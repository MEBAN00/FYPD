from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from decimal import Decimal
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")

# MongoDB Connection
MONGO_URI = "mongodb+srv://{DB_USER}:{DB_PASSWORD}@fypd.l17lq.mongodb.net/"  # Replace with your actual MongoDB connection string

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
                    "unit_price": sale_data.unit_price,
                    "total_value": sale_data.total_value,  # Changed from total_price
                    "date": sale_data.date
                }
                
                result = await sales_collection.insert_one(sale_record)
                
                return JSONResponse(content={
                    "message": "Sale created successfully",
                    "sale_id": sale_data.sale_id,
                    "updated_inventory": await inventory_collection.find_one({"product": sale_data.product})
                })
    
    except Exception as e:
        logging.error(f"Error creating sale: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating sale: {str(e)}")
@app.get("/sales-summary")
async def get_sales_summary():
    try:
        # Get current date
        current_date = datetime.now()
        today = current_date.date()
        first_day_of_month = today.replace(day=1)
        
        # Sales collection
        sales_collection = db["FYPDS"]
        
        # Aggregate today's sales
        today_sales_pipeline = [
            {
                "$match": {
                    "$expr": {
                        "$eq": [
                            {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}},
                            today.strftime("%Y-%m-%d")
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_sales": {"$sum": "$total_price"},
                    "total_transactions": {"$sum": 1},
                    "top_products": {
                        "$push": {
                            "product": "$product",
                            "quantity": "$quantity",
                            "total_price": "$total_price"
                        }
                    }
                }
            }
        ]
        
        # Aggregate monthly sales
        month_sales_pipeline = [
            {
                "$match": {
                    "$expr": {
                        "$and": [
                            {"$gte": ["$date", first_day_of_month]},
                            {"$lte": ["$date", current_date]}
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_sales": {"$sum": "$total_price"},
                    "total_transactions": {"$sum": 1},
                    "top_products": {
                        "$push": {
                            "product": "$product",
                            "quantity": "$quantity",
                            "total_price": "$total_price"
                        }
                    }
                }
            }
        ]
        
        # Find best-selling product
        best_selling_pipeline = [
            {
                "$group": {
                    "_id": "$product",
                    "total_quantity": {"$sum": "$quantity"},
                    "total_revenue": {"$sum": "$total_price"}
                }
            },
            {"$sort": {"total_quantity": -1}},
            {"$limit": 1}
        ]
        
        # Execute aggregations
        today_sales = await sales_collection.aggregate(today_sales_pipeline).to_list(length=1)
        month_sales = await sales_collection.aggregate(month_sales_pipeline).to_list(length=1)
        best_selling = await sales_collection.aggregate(best_selling_pipeline).to_list(length=1)
        
        # Prepare response
        response = {
            "today_sales": {
                "total_sales": today_sales[0]['total_sales'] if today_sales else 0,
                "total_transactions": today_sales[0]['total_transactions'] if today_sales else 0,
                "top_products": today_sales[0]['top_products'] if today_sales else []
            },
            "month_sales": {
                "total_sales": month_sales[0]['total_sales'] if month_sales else 0,
                "total_transactions": month_sales[0]['total_transactions'] if month_sales else 0,
                "top_products": month_sales[0]['top_products'] if month_sales else []
            },
            "best_selling_product": best_selling[0] if best_selling else None
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logging.error(f"Error fetching sales summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching sales summary")
# Custom exception handler for validation errors
@app.exception_handler(HTTPException)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )