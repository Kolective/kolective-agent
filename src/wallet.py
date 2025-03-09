import os
import orjson
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

class AgentWallet:
    def __init__(self):
        self.file_path = "./data/wallet.json"
        SONIC_RPC_URL = Web3.HTTPProvider(os.getenv("SONIC_RPC_URL"))
        self.w3 = Web3(SONIC_RPC_URL)
        self.admin_private_key=os.getenv("PRIVATE_KEY")

    async def create_wallet(self, user_address):
        existing_data = await self._load_existing_data()
        
        for entry in existing_data:
            if entry["user_address"] == user_address:
                print(f"Wallet already exists for user address: {user_address}")
                return
        
        private_key = self.w3.eth.account.create()._private_key.hex()
        await self.save_wallet_data(private_key, user_address)
        

    async def save_wallet_data(self, private_key, user_address):
        output_data = {
            "user_address": user_address,
            "data": private_key,
            "risk_profile": None,
            "tweet_transactions": []
        }

        existing_data = await self._load_existing_data()
        existing_data.append(output_data)
        await self._save_data(existing_data)
        print("Wallet data saved successfully.")

    async def fetch_data(self, user_address):
        existing_data = await self._load_existing_data()

        for entry in existing_data:
            if entry["user_address"] == user_address:
                private_key = entry["data"]
                
                return private_key

        print(f"No wallet data found for user address: {user_address}")
        return None
    
    async def _check_address(self, user_address):
        private_key = await self.fetch_data(user_address)
        account = Web3().eth.account.from_key(private_key)
        return account.address
    
    async def _fund_wallet(self, user_address):
        private_key = await self.fetch_data(user_address)

        sender_address = self.w3.eth.account.from_key(self.admin_private_key).address
        receiver_address = self.w3.eth.account.from_key(private_key).address
        
        nonce = self.w3.eth.get_transaction_count(sender_address)
        transaction = {
            'to': receiver_address,
            'value': self.w3.to_wei(0.5, 'ether'),
            'gas': 1000000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': nonce,
            'chainId': 57054 ,
        }

        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.admin_private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_hash.hex()
    
    async def mint(self, user_address, amount):
        amount = int(amount) * (10 ** 18)
        abi = await self._read_abi("./abi/MockToken.json")
        
        private_key = await self.fetch_data(user_address)
        sender_address = self.w3.eth.account.from_key(private_key).address
        
        token_contract = self.w3.eth.contract(address="0xEA48745C8c9eFe2e27Cc939cC5c976A54285ad5b", abi=abi)
        nonce = self.w3.eth.get_transaction_count(sender_address)
        
        transaction = token_contract.functions.mint(sender_address, amount).build_transaction({
            'chainId': 57054 ,
            'gas': 1000000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': nonce,
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_hash.hex()
    
    async def swap(self, user_address, token_in, token_out, amount):
        private_key = await self.fetch_data(user_address)
        sender_address = self.w3.eth.account.from_key(private_key).address
        
        amount_generalized = int(amount) * (10 ** 18)
        
        status = await self.approve(sender_address, private_key, token_in, amount)
        
        if status:
            abi = await self._read_abi("./abi/Kolective.json")
            
            staking_contract = self.w3.eth.contract(address="0x5AB3A2CB16B21DCFaf91CDb83EC96698FaBe6A99", abi=abi)
            nonce = self.w3.eth.get_transaction_count(sender_address, 'pending')
            
            gas_estimate = staking_contract.functions.swap(token_in, token_out, amount_generalized).estimate_gas({
                'from': sender_address
            })
            
            transaction = staking_contract.functions.swap(token_in, token_out, amount_generalized).build_transaction({
                'chainId': 57054 ,
                'gas': gas_estimate + 50000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return tx_hash.hex()
        else:
            return f"Error during transaction"
        
        
    async def swap_sell(self, user_address, token_in, token_out):
        private_key = await self.fetch_data(user_address)
        sender_address = self.w3.eth.account.from_key(private_key).address

        approve_abi = await self._read_abi("./abi/MockToken.json")
        token_contract = self.w3.eth.contract(address=token_in, abi=approve_abi)

        balance = token_contract.functions.balanceOf(sender_address).call()
        if balance == 0:
            return "Insufficient balance"

        amount_generalized = balance  

        allowance = token_contract.functions.allowance(sender_address, "0x5AB3A2CB16B21DCFaf91CDb83EC96698FaBe6A99").call()
        if allowance < amount_generalized:
            approval_status = await self.approve(sender_address, private_key, token_in, amount_generalized)
            if not approval_status:
                return "Approval failed"

        abi = await self._read_abi("./abi/Kolective.json")
        swap_contract = self.w3.eth.contract(address="0x5AB3A2CB16B21DCFaf91CDb83EC96698FaBe6A99", abi=abi)

        nonce = self.w3.eth.get_transaction_count(sender_address, 'pending')
        gas_estimate = swap_contract.functions.swap(token_in, token_out, amount_generalized).estimate_gas({
            'from': sender_address
        })

        transaction = swap_contract.functions.swap(token_in, token_out, amount_generalized).build_transaction({
            'chainId': 57054,
            'gas': gas_estimate + 50000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': nonce,
        })

        signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)

        return f"Swap successful! Tx Hash: {tx_hash.hex()}"

    
    async def approve(self, sender_address, private_key, token_in, amount):
        try:
            approve_abi = await self._read_abi("./abi/MockToken.json")
            amount = int(amount) * (10 ** 18)
            
            token_contract = self.w3.eth.contract(address=token_in, abi=approve_abi)
            nonce = self.w3.eth.get_transaction_count(sender_address)
            
            transaction = token_contract.functions.approve("0x5AB3A2CB16B21DCFaf91CDb83EC96698FaBe6A99", amount+10).build_transaction({
                'chainId': 57054 ,
                'gas': 1000000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return True
        
        except Exception as e:
            return False
    
    async def _get_balance(self, user_address, contract_address):
        abi = await self._read_abi("./abi/MockToken.json")
        private_key = await self.fetch_data(user_address)
        agent_address = self.w3.eth.account.from_key(private_key).address
        
        token_contract = self.w3.eth.contract(address=contract_address, abi=abi)
        balance = token_contract.functions.balanceOf(agent_address).call()

        return balance


    async def _read_abi(self, abi_path):
        with open(abi_path, 'r') as file:
            return orjson.loads(file.read())


    async def _load_existing_data(self):
        if not os.path.exists(self.file_path):
            return []

        with open(self.file_path, 'rb') as file:
            return orjson.loads(file.read())

    async def _save_data(self, data):
        with open(self.file_path, 'wb') as file:
            file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
