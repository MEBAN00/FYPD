from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from auth import get_current_user, db
import csv
import io
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

router = APIRouter()

async def generate_predicted_profits_csv(current_user: dict) -> str:
    """Generate CSV data for predicted gross profits"""
    try:
        predictions_collection = db["FYPD_PREDICTIONS"]
        
        # Get all predictions
        predictions = await predictions_collection.find().sort("prediction_date", -1).to_list(length=1000)
        
        if not predictions:
            # Create sample predictions data if none exist
            predictions = [
                {
                    "product": "Sample Product 1",
                    "quantity": 100,
                    "unit_price": 25.50,
                    "total_cost": 1500.00,
                    "total_price": 2550.00,
                    "predicted_profit": 1050.00,
                    "prediction_date": datetime.now() - timedelta(days=1),
                    "year": 2024,
                    "user": "sample_user"
                }
            ]
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Product', 'Quantity', 'Unit Price', 'Total Cost', 
            'Total Price', 'Predicted Profit', 'Prediction Date', 'Year', 'User'
        ])
        
        # Write data rows
        for prediction in predictions:
            writer.writerow([
                prediction.get('product', ''),
                prediction.get('quantity', 0),
                prediction.get('unit_price', 0),
                prediction.get('total_cost', 0),
                prediction.get('total_price', 0),
                prediction.get('predicted_profit', 0),
                prediction.get('prediction_date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                prediction.get('year', 2024),
                prediction.get('user', '')
            ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error generating predicted profits CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating predicted profits data")

async def generate_sales_csv(current_user: dict) -> str:
    """Generate CSV data for sales"""
    try:
        sales_collection = db["FYPDS"]
        
        # Get all sales data
        sales = await sales_collection.find().sort("date", -1).to_list(length=1000)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Sale ID', 'Product', 'Quantity', 'Unit Price', 
            'Total Value', 'Sale Date'
        ])
        
        # Write data rows
        for sale in sales:
            writer.writerow([
                sale.get('sale_id', ''),
                sale.get('product', ''),
                sale.get('quantity', 0),
                float(sale.get('unit_price', 0)),
                float(sale.get('total_value', 0)),
                sale.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error generating sales CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating sales data")

async def generate_inventory_updates_csv(current_user: dict) -> str:
    """Generate CSV data for inventory updates only"""
    try:
        # Get inventory updates from the dedicated collection
        inventory_updates_collection = db["FYPD_INVENTORY_UPDATES"]
        
        # Get all inventory updates, sorted by date in descending order
        updates = await inventory_updates_collection.find().sort("update_date", -1).to_list(length=1000)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header with enhanced fields
        writer.writerow([
            'Product', 'Update Type', 'Quantity Changed', 'Previous Stock', 
            'New Stock', 'Update Date', 'Reason', 'User', 'Additional Details'
        ])
        
        # Write data rows
        for update in updates:
            # Format additional details as a readable string
            additional_details = update.get('additional_details', {})
            details_str = '; '.join([f"{k}: {v}" for k, v in additional_details.items()]) if additional_details else ''
            
            writer.writerow([
                update.get('product', ''),
                update.get('update_type', ''),
                update.get('quantity_changed', 0),
                update.get('previous_stock', 0),
                update.get('new_stock', 0),
                update.get('update_date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                update.get('reason', ''),
                update.get('user', ''),
                details_str
            ])
        
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error generating inventory updates CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating inventory updates data")

@router.get("/download-csv/{csv_type}")
async def download_csv(csv_type: str, current_user: dict = Depends(get_current_user)):
    """Download individual CSV files"""
    try:
        if csv_type == "profits":
            csv_content = await generate_predicted_profits_csv(current_user)
            filename = f"predicted_profits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif csv_type == "sales":
            csv_content = await generate_sales_csv(current_user)
            filename = f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif csv_type == "inventory":
            csv_content = await generate_inventory_updates_csv(current_user)
            filename = f"inventory_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            raise HTTPException(status_code=400, detail="Invalid CSV type")
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logging.error(f"Error downloading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading CSV file")

@router.get("/generate-all-csvs")
async def generate_all_csvs(current_user: dict = Depends(get_current_user)):
    """Generate all three CSV files and return their data"""
    try:
        profits_csv = await generate_predicted_profits_csv(current_user)
        sales_csv = await generate_sales_csv(current_user)
        inventory_csv = await generate_inventory_updates_csv(current_user)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return {
            "profits": {
                "filename": f"predicted_profits_{timestamp}.csv",
                "content": profits_csv
            },
            "sales": {
                "filename": f"sales_data_{timestamp}.csv", 
                "content": sales_csv
            },
            "inventory": {
                "filename": f"inventory_updates_{timestamp}.csv",
                "content": inventory_csv
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating all CSVs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating CSV files")