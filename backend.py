from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from decimal import Decimal
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
import logging
from bson import ObjectId
import json
from bson.decimal128 import Decimal128
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")

# MongoDB Connection
MONGO_URI = f"mongodb+srv://{DB_USER}:{DB_PASSWORD}@fypd.l17lq.mongodb.net/{DB_NAME}?retryWrites=true&w=majority"  # Replace with your actual MongoDB connection string

# Create MongoDB client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DB_NAME]

# Load the model
try:
    dt_model = pickle.load(open("ACTIV/project_model.pkl", "rb"))
    column_transformer = pickle.load(open("ACTIV/column_transformer.pkl", "rb"))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    dt_model = None

# Create the FastAPI app
app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure expected from the frontend
class PredictionInput(BaseModel):
    product: str
    quantity: int
    unit_price: float
    total_cost: float
    total_price: float
    Year: int


class SaleInput(BaseModel):
    sale_id: str
    product: str
    quantity: int
    unit_price: Decimal   # Ensure positive decimal
    total_value: Decimal   # Changed from total_price
    date: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, Decimal128):
            return float(o.to_decimal())  # Convert Decimal128 to float
        elif isinstance(o, datetime):
            return o.isoformat()  # Convert datetime to ISO 8601 string
        return super().default(o)

@app.post("/predict")
async def predict(new_data: PredictionInput):
    if dt_model is None:
        logging.error("Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert the input data to a DataFrame
        new_df = pd.DataFrame([new_data.dict()])
        new_data_transformed = column_transformer.transform(new_df)
        pred = dt_model.predict(new_data_transformed)
        
        # Prepare response
        return JSONResponse(content={
            "prediction": int(pred[0])
        })
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction")

@app.get("/monthly-revenue")
async def get_monthly_revenue():
    try:
        # Adjust to fetch last year's data
        current_date = datetime.now()
        last_year = current_date.year - 1

        # Debugging: Log date details
        logging.info(f"Fetching data for year: {last_year}")

        # Aggregate monthly revenue for last year
        pipeline = [
            {
                "$match": {
                    "$expr": {
                        "$eq": [{"$year": "$date"}, last_year]
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_revenue": {"$sum": "$total_price"},
                    "total_transactions": {"$sum": 1}
                }
            }
        ]

        # Execute aggregation
        result = await db.FYPD.aggregate(pipeline).to_list(length=1)

        # Debugging: Log aggregation result
        # logging.info(f"Last year's monthly aggregation result: {result}")

        if result:
            monthly_data = result[0]

            # Get monthly revenue trends for the entire last year
            yearly_pipeline = [
                {
                    "$match": {
                        "$expr": {"$eq": [{"$year": "$date"}, last_year]}
                    }
                },
                {
                    "$group": {
                        "_id": {"$month": "$date"},
                        "monthly_revenue": {"$sum": "$total_price"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]

            yearly_result = await db.FYPD.aggregate(yearly_pipeline).to_list(length=12)

            # Debugging: Log yearly aggregation result
            # logging.info(f"Last year's yearly aggregation result: {yearly_result}")

            # Prepare chart data
            chart_labels = [
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ]
            chart_data = [0] * 12

            for month_data in yearly_result:
                month_index = month_data['_id'] - 1
                chart_data[month_index] = round(float(month_data['monthly_revenue'].to_decimal()), 2)

            return JSONResponse(content={
                "total_revenue": round(float(monthly_data['total_revenue'].to_decimal()), 2),
                "total_transactions": monthly_data['total_transactions'],
                "chart_labels": chart_labels,
                "chart_data": chart_data
            })

        else:
            # Debugging: Log no data found
            logging.warning("No data found for last year.")

            return JSONResponse(content={
                "total_revenue": 0,
                "total_transactions": 0,
                "chart_labels": [],
                "chart_data": []
            })


    except Exception as e:
        logging.error(f"Error fetching monthly revenue: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching monthly revenue")

@app.get("/inventory")
async def get_inventory():
    try:
        # Fetch all inventory data from MongoDB
        inventory_collection = db["FYPDI"]
        products = await inventory_collection.find().to_list(length=None)  # No limit

        # Remove duplicates by using a dictionary keyed by product name
        unique_products = {}
        for product in products:
            # Use product name as the unique key
            product_key = product.get('product', '').lower()
            if product_key not in unique_products:
                unique_products[product_key] = product

        # Convert the unique products dictionary back to a list
        unique_products_list = list(unique_products.values())

        # Sort products alphabetically by product name
        sorted_products = sorted(unique_products_list, key=lambda x: x['product'].lower())
        
        # Prepare response data
        total_inventory_value = Decimal('0')
        low_stock_products = 0
        inventory_data = []
        
        for product in sorted_products:
            # Convert Decimal128 to Decimal for calculations
            unit_price = Decimal(str(product['unit_price']))
            in_stock = int(product['in_stock'])
            
            # Calculate total value
            total_value = unit_price * Decimal(str(in_stock))
            total_inventory_value += total_value
            
            if in_stock < 50:  # Define "low stock" threshold
                low_stock_products += 1
            
            inventory_data.append({
                "product": product['product'],
                "category": product['category'],
                "in_stock": in_stock,
                "unit_price": float(unit_price),  # Convert to float for JSON serialization
                "total_value": float(total_value)  # Convert to float for JSON serialization
            })
        
        # Total product count after deduplication
        total_products = len(sorted_products)

        response = {
            "inventory": inventory_data,
            "total_products": total_products,
            "total_inventory_value": float(total_inventory_value),  # Convert to float for JSON serialization
            "low_stock_products": low_stock_products
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        logging.error(f"Error fetching inventory: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching inventory data")

    
@app.post("/add-product")
async def add_product(product_data: dict):
    try:
        # Validate input data
        required_fields = ['product', 'category', 'in_stock', 'unit_price']
        for field in required_fields:
            if field not in product_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Prepare the product document
        new_product = {
            'product': product_data['product'],
            'category': product_data['category'],
            'in_stock': int(product_data['in_stock']),
            'unit_price': float(product_data['unit_price'])  # Convert to float instead of Decimal
        }
        
        # Insert into MongoDB
        inventory_collection = db["FYPDI"]
        result = await inventory_collection.insert_one(new_product)
        
        # Prepare response
        return JSONResponse(content={
            "message": "Product added successfully",
            "product_id": str(result.inserted_id)
        })
    
    except Exception as e:
        logging.error(f"Error adding product: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding product: {str(e)}")
    

@app.post("/create-sale")
async def create_sale(sale_data: SaleInput):
    try:
        # Start a database transaction
        async with await mongo_client.start_session() as session:
            async with session.start_transaction():
                # 1. Update Inventory
                inventory_collection = db["FYPDI"]
                
                # Find the product in inventory
                product = await inventory_collection.find_one({"product": sale_data.product})
                
                if not product:
                    raise HTTPException(status_code=404, detail="Product not found")
                
                # Check if sufficient stock
                if product['in_stock'] < sale_data.quantity:
                    raise HTTPException(status_code=400, detail="Insufficient inventory")
                
                # Reduce inventory
                await inventory_collection.update_one(
                    {"product": sale_data.product},
                    {"$inc": {"in_stock": -sale_data.quantity}}
                )
                
                # 2. Record Sale in Sales Collection
                sales_collection = db["FYPDS"]
                sale_record = {
                    "sale_id": sale_data.sale_id,
                    "product": sale_data.product,
                    "quantity": sale_data.quantity,
                    "unit_price": float(sale_data.unit_price),
                    "total_value": float(sale_data.total_value),  # Changed from total_price
                    "date": sale_data.date
                }
                
                result = await sales_collection.insert_one(sale_record)

                # When fetching the updated inventory, convert ObjectId to string
                updated_inventory = await inventory_collection.find_one({"product": sale_data.product})

                # Use json.loads and json.dumps with the custom encoder
                serialized_inventory = json.loads(json.dumps(updated_inventory, cls=MongoJSONEncoder))
                
                return JSONResponse(content={
                    "message": "Sale created successfully",
                    "sale_id": sale_data.sale_id,
                    "updated_inventory": serialized_inventory
                })

    
    except Exception as e:
        logging.error(f"Error creating sale: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating sale: {str(e)}")@app.get("/sales-summary")

@app.get("/sales-summary")
async def get_sales_summary():
    try:
        # Get current date
        current_date = datetime.now()
        today_start = datetime.combine(current_date.date(), datetime.min.time())
        today_end = datetime.combine(current_date.date(), datetime.max.time())
        
        # Calculate month start and end
        month_start = datetime.combine(current_date.replace(day=1), datetime.min.time())
        month_end = datetime.combine(current_date, datetime.max.time())
        
        # Sales collection
        sales_collection = db["FYPDS"]
        
        # Aggregate today's sales with proper date comparison
        today_sales_pipeline = [
            {
                "$match": {
                    "date": {
                        "$gte": today_start,
                        "$lte": today_end
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_sales": {"$sum": "$total_value"},
                    "total_transactions": {"$sum": 1}
                }
            }
        ]
        
        # Find best-selling product of all time
        best_selling_pipeline = [
            {
                "$group": {
                    "_id": "$product",
                    "total_quantity": {"$sum": "$quantity"},
                    "total_revenue": {"$sum": "$total_value"}
                }
            },
            {"$sort": {"total_quantity": -1}},
            {"$limit": 1}
        ]
        
        # Aggregate this month's sales
        month_sales_pipeline = [
            {
                "$match": {
                    "date": {
                        "$gte": month_start,
                        "$lte": month_end
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_sales": {"$sum": "$total_value"},
                    "total_transactions": {"$sum": 1}
                }
            }
        ]
        
        # Execute aggregations
        today_sales = await sales_collection.aggregate(today_sales_pipeline).to_list(length=1)
        month_sales = await sales_collection.aggregate(month_sales_pipeline).to_list(length=1)
        best_selling = await sales_collection.aggregate(best_selling_pipeline).to_list(length=1)
        
        # Prepare response
        response = {
            "today_sales": {
                "total_sales": float(today_sales[0]['total_sales']) if today_sales else 0,
                "total_transactions": today_sales[0]['total_transactions'] if today_sales else 0
            },
            "month_sales": {
                "total_sales": float(month_sales[0]['total_sales']) if month_sales else 0,
                "total_transactions": month_sales[0]['total_transactions'] if month_sales else 0
            },
            "best_selling_product": {
                "product": best_selling[0]['_id'],
                "total_quantity": best_selling[0]['total_quantity'],
                "total_revenue": float(best_selling[0]['total_revenue'])
            } if best_selling else None
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logging.error(f"Error fetching sales summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching sales summary")
    
@app.get("/sales")
async def get_sales():
    try:
        # Sales collection
        sales_collection = db["FYPDS"]
        
        # Fetch sales, sorted by date in descending order
        sales_pipeline = [
            {"$sort": {"date": -1}},  # Sort by date, most recent first
            {"$limit": 100}  # Limit to last 100 sales to prevent overwhelming the frontend
        ]
        
        # Execute aggregation
        sales = await sales_collection.aggregate(sales_pipeline).to_list(length=None)
        
        # Convert ObjectId, Decimal128, and datetime to JSON-serializable format
        serialized_sales = json.loads(json.dumps(sales, cls=MongoJSONEncoder))
        
        return JSONResponse(content=serialized_sales)
    
    except Exception as e:
        logging.error(f"Error fetching sales: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching sales data")
    
# Add these new endpoints to your backend.py file

@app.put("/update-product/{product_id}")
async def update_product(product_id: str, product_data: dict):
    try:
        # Convert product_id to ObjectId if necessary
        inventory_collection = db["FYPDI"]
        
        # Validate input data
        update_fields = {}
        if 'product' in product_data:
            update_fields['product'] = product_data['product']
        if 'category' in product_data:
            update_fields['category'] = product_data['category']
        if 'in_stock' in product_data:
            update_fields['in_stock'] = int(product_data['in_stock'])
        if 'unit_price' in product_data:
            update_fields['unit_price'] = float(product_data['unit_price'])
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No update fields provided")
        
        # Perform the update
        result = await inventory_collection.update_one(
            {"product": product_id},  # Use product name as identifier
            {"$set": update_fields}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Fetch the updated product to return
        updated_product = await inventory_collection.find_one({"product": product_id})
        
        # Convert ObjectId, Decimal128 to JSON-serializable format
        serialized_product = json.loads(json.dumps(updated_product, cls=MongoJSONEncoder))
        
        return JSONResponse(content=serialized_product)
    
    except Exception as e:
        logging.error(f"Error updating product: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating product: {str(e)}")

@app.delete("/delete-product/{product_name}")
async def delete_product(product_name: str):
    try:
        inventory_collection = db["FYPDI"]
        
        # Delete the product
        result = await inventory_collection.delete_one({"product": product_name})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return JSONResponse(content={
            "message": "Product deleted successfully",
            "product_name": product_name
        })
    
    except Exception as e:
        logging.error(f"Error deleting product: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting product: {str(e)}")
# Custom exception handler for validation errors
@app.exception_handler(HTTPException)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )