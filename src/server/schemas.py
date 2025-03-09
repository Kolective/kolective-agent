from typing import Optional
from pydantic import BaseModel

class QueryRequestClassifier(BaseModel):
    data: str
    user_address: str
    
class QueryRequestRecommendation(BaseModel):
    data: str
    user_address: str
    
class QueryUserWallet(BaseModel):
    user_address: str
    
class QueryMint(BaseModel):
    user_address: str
    amount: str

class QuerySwap(BaseModel):
    user_address: str
    token_in: str
    token_out: str
    amount: str
    
    
async def get_address_data(key=None):
    addresses = {
        "TokenFactory": "0xF17a1EA23F0a943c2016b5F2b8f656Ada273a23c",
        "Core": "0x5AB3A2CB16B21DCFaf91CDb83EC96698FaBe6A99",
        "SONIC": "0xEA48745C8c9eFe2e27Cc939cC5c976A54285ad5b",
        "Ethereum": "0x5ffac3827ddEE20b479d27a2b72c5a44FcD752DF",
        "Bitcoin": "0x3e2DA21A9Cc78C7401cAF4FaA85de092ED726833",
        "Wrapped_Ether": "0x092eDa418B2f2fEeb86746206a8dA9c7a4c4Fa95",
        "PEPE": "0xe66022A565E01A4Db313056e9f0D061A76020675",
        "OFFICIAL_TRUMP": "0xb589fad21Abb69b2c7A6ADD0909038658e086cB9",
        "DOGE_AI": "0x50Ec6cC6ACF8784093E624C8949937C406F9F808",
        "dogwifhat": "0x6BB2620f3967c277B446E83d074767f18c4e1C58",
        "STONKS": "0x6C45a072fAc0d9C08F5bf8f392818C5eb64afcFA"
    }

    if key:
        return addresses.get(key, f"Error: '{key}' not found")
    return addresses


    