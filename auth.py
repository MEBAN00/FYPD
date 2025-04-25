from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

# Copy these from your main.py
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


# Pydantic models
class Token (BaseModel):
    access_token:str
    token_type:str


class TokenData(BaseModel):
    username: Optional[str] = None

# Blacklisted tokens set
blacklisted_tokens = set()

# Token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Database connection (import this from your main.py or create a separate db.py)


DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")

# MongoDB Connection
MONGO_URI = f"mongodb+srv://{DB_USER}:{DB_PASSWORD}@fypd.l17lq.mongodb.net/{DB_NAME}?retryWrites=true&w=majority"

# Create MongoDB client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DB_NAME]

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        
        # Check if token is blacklisted
        if token in blacklisted_tokens:
            raise HTTPException(
                status_code=401,
                detail="Token has been invalidated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        token_data = TokenData(username=username)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await db.FYPDU.find_one({"name": token_data.username})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Add this function to your auth.py file

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    return current_user