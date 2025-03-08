import os
import sys
import orjson
import asyncio
import requests
from collections import defaultdict, Counter

# Menambahkan src ke sys.path agar bisa mengimpor module dengan benar
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.wallet import AgentWallet
from src.server.schemas import get_address_data

agent = AgentWallet()

# Fungsi untuk membaca user dari wallet.json
async def get_user():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wallet_dir = os.path.normpath(os.path.join(current_dir, '../../data'))
    file_path = os.path.join(wallet_dir, 'wallet.json')

    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan.")
        return []

    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())

    return [(entry['user_address'], entry['data'], entry['risk_profile']) for entry in wallet_data]

async def parse_contract_address(action: str):
    user_list = await get_user()
    list_ca = []

    for user_address, _, user_risk in user_list:
        response = requests.get(f"https://kolective-backend.vercel.app/api/kol/followed/{user_address}")
        if response.status_code == 200:
            data = response.json()

            tweets = data.get('followedKOL', {}).get('kol', {}).get('tweets', [])
            for tweet in tweets:
                if not tweet.get('expired', True) and tweet.get('signal').lower() == action and tweet.get('risk').lower() == user_risk:
                    list_ca.append((user_address, tweet['token']['addressToken'], tweet['id']))
                    
    return list_ca

async def _check_tweet_transaction(user_address: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wallet_dir = os.path.normpath(os.path.join(current_dir, '../../data'))
    file_path = os.path.join(wallet_dir, 'wallet.json')
    
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())

    for entry in wallet_data:
        if entry["user_address"] == user_address:
            return entry.get("tweet_transactions", [])
    
    return []


async def _update_tweet_transaction(user_address: str, tweet_id: int):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wallet_dir = os.path.normpath(os.path.join(current_dir, '../../data'))
    file_path = os.path.join(wallet_dir, 'wallet.json')

    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan.")
        return []

    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())
    
    for entry in wallet_data:
        if entry["user_address"] == user_address:
            entry["tweet_transactions"].append(tweet_id)
            break
    
    with open(file_path, 'wb') as file:
        file.write(orjson.dumps(wallet_data, option=orjson.OPT_INDENT_2))
        
        

async def append_allocation(data):
    user_contracts = defaultdict(list)
    tweet_ids = [tweet_id for _, _, tweet_id in data]
    
    for user, contract,_ in data:
        user_contracts[user].append(contract)

    allocation = {}
    for user, contracts in user_contracts.items():
        contract_count = Counter(contracts)
        total_contracts = sum(contract_count.values())
        allocation[user] = {contract: (count / total_contracts) * 100 for contract, count in contract_count.items()}

    return allocation, tweet_ids


async def get_balance(address, contract_address):
    try:
        balance = await agent._get_balance(address, contract_address)
        return balance if balance is not None else 0
    except Exception as e:
        print(f"Error fetching balance for {address}: {e}")
        return 0

# Main function
async def transactions(action: str):
    while True:
        list_ca = await parse_contract_address(action)
        allocation, tweet_ids = await append_allocation(list_ca)
        
        for user_address, contract_allocations in allocation.items():
            processed_tweets = await _check_tweet_transaction(user_address)
            
            match action:
                case 'buy':
                    balance = await get_balance(user_address, await get_address_data('SONIC'))
                    
                    if balance > 0:
                        allocated_balances = {
                            contract: ((balance * percentage / 100) // (10 ** 18)) for contract, percentage in contract_allocations.items()
                        }
                        
                        for (contract, allocated_amount), tweet_id in zip(allocated_balances.items(), tweet_ids):
                            if tweet_id in processed_tweets:
                                print(f"Skipping tweet_id {tweet_id}, already processed.")
                                continue
                            
                            print(f"  Contract {contract}: {allocated_amount:.2f}")
                            print("Executing transaction...")
                            
                            tx_hash = await agent.swap_buy(
                                user_address=user_address,
                                token_in=await get_address_data('SONIC'),
                                token_out=contract,
                                amount=allocated_amount
                            )
                            print(f"Successful Buy Trade! txhash: {tx_hash}")
                            
                            await _update_tweet_transaction(user_address, tweet_id)

                case 'sell':
                    processed_tweets = await _check_tweet_transaction(user_address)
                    
                    if processed_tweets:
                        for contract, _ in contract_allocations.items():
                            token_balance = await get_balance(user_address, contract)
                            if token_balance > 0:
                                tx_hash = await agent.swap_sell(
                                    user_address=user_address,
                                    token_in=contract,
                                    token_out=await get_address_data('SONIC')
                                )
                                print(f"Successful Sell Trade! txhash: {tx_hash}")
                    else:
                        print(f"No previous buy transactions for {user_address}. Skipping transaction.")

        await asyncio.sleep(5)
        

# Testing transaction
# asyncio.run(transactions())
