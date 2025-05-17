from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from decimal import Decimal
import pandas as pd
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pickle
import os
from dotenv import load_dotenv
import logging
from bson import ObjectId
import json
from bson.decimal128 import Decimal128
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from passlib.hash import bcrypt
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from typing import Optional
from fastapi import Cookie, Response, Request
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from admin_routes import router as admin_router

from auth import get_current_user, db, mongo_client, create_access_token, blacklisted_tokens, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

load_dotenv()

security = HTTPBearer()

# Initialize logging
logging.basicConfig(level=logging.INFO)

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")




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

class UserSignup(BaseModel):
    name: str
    business_name: str
    dob: date
    password: str

class UserLogin(BaseModel):
    name: str
    password: str

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

# Add this line to include the admin routes
app.include_router(admin_router)

@app.post("/predict")
async def predict(new_data: PredictionInput, current_user: dict = Depends(get_current_user)):
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
async def get_monthly_revenue(current_user: dict = Depends(get_current_user)):
    try:
        # Adjust to fetch last year's data
        current_date = datetime.now()
        last_year = current_date.year - 2

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
async def get_inventory(current_user: dict = Depends(get_current_user)):
    try:
        # Fetch all inventory data from MongoDB
        inventory_collection = db["FYPDI"]
        products = await inventory_collection.find().to_list(length=50)  # No limit

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
async def add_product(product_data: dict, current_user: dict = Depends(get_current_user)):
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
async def create_sale(sale_data: SaleInput, current_user: dict = Depends(get_current_user)):
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
async def get_sales_summary(current_user: dict = Depends(get_current_user)):
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
async def get_sales(current_user: dict = Depends(get_current_user)):
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
    
@app.put("/update-product/{product_id}")
async def update_product(product_id: str, product_data: dict, current_user: dict = Depends(get_current_user)):
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
async def delete_product(product_name: str, current_user: dict = Depends(get_current_user)):
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

@app.get("/inventory-turnover")
async def get_inventory_turnover(current_user: dict = Depends(get_current_user)):
    try:
        # Get current date
        current_date = datetime.now()
        
        # Calculate year start
        year_start = datetime(current_date.year, 1, 1)
        
        # Get sales collection and inventory collection
        sales_collection = db["FYPDS"]
        inventory_collection = db["FYPDI"]
        
        # Limit inventory fetch to 50 products
        inventory = await inventory_collection.find().sort("product", 1).limit(50).to_list(length=50)
        
        # Get the product names from the limited inventory
        product_names = [product["product"] for product in inventory]
        
        # Calculate sales for only the selected products
        sales_pipeline = [
            {
                "$match": {
                    "date": {
                        "$gte": year_start,
                        "$lte": current_date
                    },
                    "product": {"$in": product_names}  # Only match products in our limited inventory
                }
            },
            {
                "$group": {
                    "_id": "$product",
                    "total_quantity_sold": {"$sum": "$quantity"},
                    "total_sales_value": {"$sum": "$total_value"}
                }
            }
        ]
        
        sales_data = await sales_collection.aggregate(sales_pipeline).to_list(length=None)
        
        # Calculate turnover metrics
        turnover_data = []
        for product in inventory:
            product_sales = next((item for item in sales_data if item["_id"] == product["product"]), None)
            
            # Calculate months elapsed in current year
            months_elapsed = (current_date - year_start).days / 30.44  # Average days per month
            
            if product_sales:
                # Calculate annual turnover rate (annualized based on current year data)
                annual_turnover_rate = (product_sales["total_quantity_sold"] / product["in_stock"]) * (12 / months_elapsed) if product["in_stock"] > 0 else 0
                
                # Calculate days inventory outstanding (DIO)
                if product_sales["total_quantity_sold"] > 0:
                    daily_sales = product_sales["total_quantity_sold"] / ((current_date - year_start).days)
                    days_inventory = product["in_stock"] / daily_sales if daily_sales > 0 else float('inf')
                else:
                    days_inventory = float('inf')
                
                turnover_data.append({
                    "product": product["product"],
                    "current_stock": product["in_stock"],
                    "quantity_sold": product_sales["total_quantity_sold"],
                    "turnover_rate": round(annual_turnover_rate, 2),
                    "days_inventory": round(days_inventory, 1),
                    "sales_value": float(product_sales["total_sales_value"])
                })
            else:
                # Include products with no sales
                turnover_data.append({
                    "product": product["product"],
                    "current_stock": product["in_stock"],
                    "quantity_sold": 0,
                    "turnover_rate": 0,
                    "days_inventory": float('inf'),
                    "sales_value": 0
                })
        
        # Sort by turnover rate descending
        turnover_data.sort(key=lambda x: x["turnover_rate"], reverse=True)
        
        # Calculate overall metrics for the limited set
        total_inventory = sum(item["current_stock"] for item in turnover_data)
        total_sales = sum(item["quantity_sold"] for item in turnover_data)
        
        if total_inventory > 0:
            overall_turnover = (total_sales / total_inventory) * (12 / months_elapsed)
        else:
            overall_turnover = 0
        
        response = {
            "turnover_data": turnover_data,
            "overall_metrics": {
                "overall_turnover_rate": round(overall_turnover, 2),
                "total_inventory": total_inventory,
                "total_sales": total_sales
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logging.error(f"Error calculating inventory turnover: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating inventory turnover")
    
@app.post("/signup")
async def signup(user_data: UserSignup):
    try:
        # Check if user already exists
        users_collection = db["FYPDU"]
        existing_user = await users_collection.find_one({"name": user_data.name})
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash the password
        hashed_password = bcrypt.hash(user_data.password)

        # Convert date to datetime for MongoDB compatibility
        dob_datetime = datetime.combine(user_data.dob, datetime.min.time())
        
        # Create user document
        user_doc = {
            "name": user_data.name,
            "business_name": user_data.business_name,
            "dob": dob_datetime,
            "password": hashed_password
        }
        
        # Insert into database
        result = await users_collection.insert_one(user_doc)
        
        return JSONResponse(content={
            "message": "User created successfully",
            "user_id": str(result.inserted_id)
        })
        
    except Exception as e:
        logging.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(user_data: UserLogin):
    try:
        users_collection = db["FYPDU"]
        user = await users_collection.find_one({"name": user_data.name})
        
        if not user or not bcrypt.verify(user_data.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")  
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["name"]}, expires_delta=access_token_expires
        )
        
        # Make sure to include the is_admin field from the database
        # Use .get() with a default of False to handle cases where the field might not exist
        is_admin = user.get("is_admin", False)
        
        # Debug log to check the admin status
        print(f"Is admin: {is_admin}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "name": user["name"],
            "business_name": user["business_name"],
            "is_admin": is_admin  # Include the correct admin status
        }
        
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        
        # Option 1: Simple in-memory blacklisting
        blacklisted_tokens.add(token)
        
        # Option 2: MongoDB blacklisting (more persistent)
        # await db.blacklisted_tokens.insert_one({
        #     "token": token,
        #     "blacklisted_at": datetime.utcnow()
        # })
        
        return {"message": "Successfully logged out"}
    except Exception as e:
        logging.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Custom exception handler for validation errors
@app.exception_handler(HTTPException)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )