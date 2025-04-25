from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List, Optional
import csv
import io
import json
import logging
from datetime import datetime, timedelta
from bson import ObjectId
import os
import shutil

# Import from auth.py instead of main.py
from auth import get_current_user, get_admin_user, db, mongo_client

# Create router
router = APIRouter(prefix="/admin", tags=["admin"])

# Helper function to check if user is admin
async def get_admin_user(current_user: dict = Depends(get_admin_user)):
    # You might want to add an admin field to your user model
    # For now, we'll just check if the user exists
    if not current_user:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# User Management Endpoints
@router.get("/users")
async def get_all_users(admin_user: dict = Depends(get_admin_user)):
    try:
        users = await db.FYPDU.find().to_list(length=100)
        # Convert ObjectId to string for JSON serialization
        for user in users:
            user["_id"] = str(user["_id"])
            # Don't send password hash
            if "password" in user:
                del user["password"]
        return users
    except Exception as e:
        logging.error(f"Error fetching users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")

@router.post("/users")
async def create_user(user_data: dict, admin_user: dict = Depends(get_admin_user)):
    try:
        # Validate required fields
        required_fields = ["name", "business_name", "dob", "password"]
        for field in required_fields:
            if field not in user_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Check if user already exists
        existing_user = await db.FYPDU.find_one({"name": user_data["name"]})
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash the password
        from main import bcrypt
        hashed_password = bcrypt.hash(user_data["password"])
        
        # Convert date string to datetime
        from datetime import datetime
        dob_datetime = datetime.fromisoformat(user_data["dob"])
        
        # Create user document
        user_doc = {
            "name": user_data["name"],
            "business_name": user_data["business_name"],
            "dob": dob_datetime,
            "password": hashed_password,
            "created_at": datetime.now()
        }
        
        # Insert into database
        result = await db.FYPDU.insert_one(user_doc)
        
        return JSONResponse(content={
            "message": "User created successfully",
            "user_id": str(result.inserted_id)
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin_user: dict = Depends(get_admin_user)):
    try:
        # Convert string ID to ObjectId
        object_id = ObjectId(user_id)
        
        # Delete the user
        result = await db.FYPDU.delete_one({"_id": object_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return JSONResponse(content={
            "message": "User deleted successfully"
        })
    except Exception as e:
        logging.error(f"Error deleting user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

# Inventory Management Endpoints
@router.post("/bulk-import")
async def bulk_import_products(file: UploadFile = File(...), admin_user: dict = Depends(get_admin_user)):
    try:
        # Read CSV file
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        # Prepare products for insertion
        products = []
        for row in csv_reader:
            product = {
                "product": row.get("product"),
                "category": row.get("category"),
                "in_stock": int(row.get("in_stock", 0)),
                "unit_price": float(row.get("unit_price", 0))
            }
            products.append(product)
        
        # Insert products into database
        if products:
            result = await db.FYPDI.insert_many(products)
            
            return JSONResponse(content={
                "message": "Products imported successfully",
                "imported_count": len(result.inserted_ids)
            })
        else:
            raise HTTPException(status_code=400, detail="No valid products found in CSV")
    except Exception as e:
        logging.error(f"Error importing products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error importing products: {str(e)}")

@router.get("/export-inventory")
async def export_inventory(admin_user: dict = Depends(get_admin_user)):
    try:
        # Fetch all inventory data
        inventory = await db.FYPDI.find().to_list(length=None)
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["product", "category", "in_stock", "unit_price"])
        
        # Write data
        for item in inventory:
            writer.writerow([
                item.get("product", ""),
                item.get("category", ""),
                item.get("in_stock", 0),
                item.get("unit_price", 0)
            ])
        
        # Create response
        response = StreamingResponse(
            iter([output.getvalue()]), 
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=inventory_export.csv"
        
        return response
    except Exception as e:
        logging.error(f"Error exporting inventory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting inventory: {str(e)}")

# System Management Endpoints
@router.get("/backup")
async def backup_database(admin_user: dict = Depends(get_admin_user)):
    try:
        # Create backup directory if it doesn't exist
        os.makedirs("backups", exist_ok=True)
        
        # Get current date for filename
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backups/backup_{date_str}.json"
        
        # Fetch data from collections
        users = await db.FYPDU.find().to_list(length=None)
        inventory = await db.FYPDI.find().to_list(length=None)
        sales = await db.FYPDS.find().to_list(length=None)
        
        # Convert ObjectId to string for JSON serialization
        for collection in [users, inventory, sales]:
            for item in collection:
                item["_id"] = str(item["_id"])
        
        # Create backup data
        backup_data = {
            "users": users,
            "inventory": inventory,
            "sales": sales,
            "backup_date": datetime.now().isoformat()
        }
        
        # Write to file
        with open(backup_file, "w") as f:
            json.dump(backup_data, f)
        
        # Return file as download
        return FileResponse(
            path=backup_file,
            filename=f"database_backup_{date_str}.json",
            media_type="application/json"
        )
    except Exception as e:
        logging.error(f"Error creating backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating backup: {str(e)}")

@router.post("/restore")
async def restore_database(file: UploadFile = File(...), admin_user: dict = Depends(get_admin_user)):
    try:
        # Read backup file
        contents = await file.read()
        backup_data = json.loads(contents)
        
        # Validate backup data
        required_collections = ["users", "inventory", "sales"]
        for collection in required_collections:
            if collection not in backup_data:
                raise HTTPException(status_code=400, detail=f"Invalid backup file: missing {collection} collection")
        
        # Start a transaction
        async with await mongo_client.start_session() as session:
            async with session.start_transaction():
                # Clear existing data
                await db.FYPDU.delete_many({})
                await db.FYPDI.delete_many({})
                await db.FYPDS.delete_many({})
                
                # Convert string IDs back to ObjectId
                for collection_name in required_collections:
                    for item in backup_data[collection_name]:
                        if "_id" in item:
                            item["_id"] = ObjectId(item["_id"])
                
                # Restore data
                if backup_data["users"]:
                    await db.FYPDU.insert_many(backup_data["users"])
                if backup_data["inventory"]:
                    await db.FYPDI.insert_many(backup_data["inventory"])
                if backup_data["sales"]:
                    await db.FYPDS.insert_many(backup_data["sales"])
        
        return JSONResponse(content={
            "message": "Database restored successfully",
            "restore_date": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error restoring database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error restoring database: {str(e)}")

@router.get("/logs")
async def get_system_logs(admin_user: dict = Depends(get_admin_user)):
    try:
        # Check if log file exists
        log_file = "app.log"  # Adjust to your actual log file path
        if not os.path.exists(log_file):
            return "No logs found."
        
        # Read last 100 lines of log file
        with open(log_file, "r") as f:
            lines = f.readlines()
            last_lines = lines[-100:] if len(lines) > 100 else lines
        
        return "".join(last_lines)
    except Exception as e:
        logging.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")

# Reports Endpoints
@router.post("/reports/{report_type}")
async def generate_report(
    report_type: str, 
    params: dict, 
    background_tasks: BackgroundTasks,
    admin_user: dict = Depends(get_admin_user)
):
    try:
        # Validate report type
        valid_report_types = ["sales", "inventory", "user-activity"]
        if report_type not in valid_report_types:
            raise HTTPException(status_code=400, detail=f"Invalid report type: {report_type}")
        
        # Validate parameters
        required_params = ["start_date", "end_date", "format"]
        for param in required_params:
            if param not in params:
                raise HTTPException(status_code=400, detail=f"Missing required parameter: {param}")
        
        # Parse dates
        try:
            start_date = datetime.fromisoformat(params["start_date"])
            end_date = datetime.fromisoformat(params["end_date"])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
        
        # Validate format
        valid_formats = ["pdf", "csv", "excel"]
        if params["format"] not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Invalid format: {params['format']}")
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Generate report based on type
        if report_type == "sales":
            # Fetch sales data
            sales_pipeline = [
                {
                    "$match": {
                        "date": {
                            "$gte": start_date,
                            "$lte": end_date
                        }
                    }
                },
                {"$sort": {"date": -1}}
            ]
            
            sales = await db.FYPDS.aggregate(sales_pipeline).to_list(length=None)
            
            # Generate CSV report
            if params["format"] == "csv":
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(["Sale ID", "Product", "Quantity", "Unit Price", "Total Value", "Date"])
                
                # Write data
                for sale in sales:
                    writer.writerow([
                        sale.get("sale_id", ""),
                        sale.get("product", ""),
                        sale.get("quantity", 0),
                        sale.get("unit_price", 0),
                        sale.get("total_value", 0),
                        sale.get("date", "").isoformat() if isinstance(sale.get("date"), datetime) else sale.get("date", "")
                    ])
                
                # Create response
                response = StreamingResponse(
                    iter([output.getvalue()]), 
                    media_type="text/csv"
                )
                response.headers["Content-Disposition"] = f"attachment; filename=sales_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
                
                return response
            
            # For PDF and Excel, we'd need additional libraries
            # This is a simplified example
            return JSONResponse(content={
                "message": f"Generated {report_type} report in {params['format']} format",
                "data_count": len(sales)
            })
            
        elif report_type == "inventory":
            # Fetch inventory data
            inventory = await db.FYPDI.find().to_list(length=None)
            
            # Generate CSV report
            if params["format"] == "csv":
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(["Product", "Category", "In Stock", "Unit Price", "Total Value"])
                
                # Write data
                for item in inventory:
                    total_value = item.get("in_stock", 0) * item.get("unit_price", 0)
                    writer.writerow([
                        item.get("product", ""),
                        item.get("category", ""),
                        item.get("in_stock", 0),
                        item.get("unit_price", 0),
                        total_value
                    ])
                
                # Create response
                response = StreamingResponse(
                    iter([output.getvalue()]), 
                    media_type="text/csv"
                )
                response.headers["Content-Disposition"] = f"attachment; filename=inventory_report_{datetime.now().strftime('%Y%m%d')}.csv"
                
                return response
            
            # For PDF and Excel, we'd need additional libraries
            return JSONResponse(content={
                "message": f"Generated {report_type} report in {params['format']} format",
                "data_count": len(inventory)
            })
            
        elif report_type == "user-activity":
            # This would require a user activity log collection
            # For this example, we'll return a placeholder
            return JSONResponse(content={
                "message": f"Generated {report_type} report in {params['format']} format",
                "note": "User activity tracking not implemented yet"
            })
            
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

# Purchase Order Endpoints
@router.get("/purchase-orders")
async def get_purchase_orders(admin_user: dict = Depends(get_admin_user)):
    try:
        # Create purchase orders collection if it doesn't exist
        if "FYPPO" not in await db.list_collection_names():
            await db.create_collection("FYPPO")
        
        # Fetch purchase orders
        purchase_orders = await db.FYPPO.find().sort("created_at", -1).to_list(length=100)
        
        # Convert ObjectId to string for JSON serialization
        for po in purchase_orders:
            po["_id"] = str(po["_id"])
        
        return purchase_orders
    except Exception as e:
        logging.error(f"Error fetching purchase orders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching purchase orders: {str(e)}")

@router.post("/purchase-orders")
async def create_purchase_order(po_data: dict, admin_user: dict = Depends(get_admin_user)):
    try:
        # Validate required fields
        required_fields = ["supplier", "product", "quantity"]
        for field in required_fields:
            if field not in po_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Generate PO number
        po_count = await db.FYPPO.count_documents({})
        po_number = f"PO-{datetime.now().strftime('%Y%m%d')}-{po_count + 1:04d}"
        
        # Create PO document
        po_doc = {
            "po_number": po_number,
            "supplier": po_data["supplier"],
            "product": po_data["product"],
            "quantity": int(po_data["quantity"]),
            "status": "pending",
            "created_at": datetime.now(),
            "created_by": admin_user["name"]
        }
        
        # Insert into database
        result = await db.FYPPO.insert_one(po_doc)
        
        return JSONResponse(content={
            "message": "Purchase order created successfully",
            "po_id": str(result.inserted_id),
            "po_number": po_number
        })
    except Exception as e:
        logging.error(f"Error creating purchase order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating purchase order: {str(e)}")

@router.put("/purchase-orders/{po_id}/status")
async def update_purchase_order_status(
    po_id: str, 
    status_data: dict, 
    admin_user: dict = Depends(get_admin_user)
):
    try:
        # Validate status
        if "status" not in status_data:
            raise HTTPException(status_code=400, detail="Missing status field")
        
        valid_statuses = ["pending", "approved", "shipped", "received", "cancelled"]
        if status_data["status"] not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_data['status']}")
        
        # Convert string ID to ObjectId
        object_id = ObjectId(po_id)
        
        # Update PO status
        result = await db.FYPPO.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "status": status_data["status"],
                    "updated_at": datetime.now(),
                    "updated_by": admin_user["name"]
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Purchase order not found")
        
        # If status is "received", update inventory
        if status_data["status"] == "received":
            # Get the PO details
            po = await db.FYPPO.find_one({"_id": object_id})
            
            if po:
                # Update inventory
                await db.FYPDI.update_one(
                    {"product": po["product"]},
                    {"$inc": {"in_stock": po["quantity"]}}
                )
        
        return JSONResponse(content={
            "message": "Purchase order status updated successfully"
        })
    except Exception as e:
        logging.error(f"Error updating purchase order status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating purchase order status: {str(e)}")

@router.delete("/purchase-orders/{po_id}")
async def delete_purchase_order(po_id: str, admin_user: dict = Depends(get_admin_user)):
    try:
        # Convert string ID to ObjectId
        object_id = ObjectId(po_id)
        
        # Delete the purchase order
        result = await db.FYPPO.delete_one({"_id": object_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Purchase order not found")
        
        return JSONResponse(content={
            "message": "Purchase order deleted successfully"
        })
    except Exception as e:
        logging.error(f"Error deleting purchase order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting purchase order: {str(e)}")