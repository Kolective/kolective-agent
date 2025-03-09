import os
import sys
import orjson
import asyncio
import requests
from collections import defaultdict, Counter

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.wallet import AgentWallet
from src.server.schemas import get_address_data

agent = AgentWallet()

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
    
async def query_graphql(tx_hash):
    transport = AIOHTTPTransport(url="http://localhost:42069")
    async with Client(transport=transport, fetch_schema_from_transport=True) as client:
        query = gql("""
        query MyQuery($hash: String!) {
            swaps(where: { transactionHash: $hash }) {
                items {
                sellPrice
                sender
                transactionHash
                }
            }
        }
        """)

        variables = {"hash": tx_hash}
        result = await client.execute(query, variable_values=variables)
        print(result)
        buy_price = result['swaps']['items'][0]['sellPrice']
        return buy_price


async def _update_user_transactions(user_address, contract_address, price):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wallet_dir = os.path.normpath(os.path.join(current_dir, '../../data'))
    file_path = os.path.join(wallet_dir, 'transactions.json')
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as file:
            try:
                wallet_data = orjson.loads(file.read())
            except orjson.JSONDecodeError:
                wallet_data = {"users": []}
    else:
        wallet_data = {"users": []}
    
    user_found = False
    for user in wallet_data['users']:
        if user['user_address'] == user_address:
            user_found = True
            user['contracts'].append({
                'contract_address': contract_address,
                'price': price
            })
            break
    
    if not user_found:
        wallet_data['users'].append({
            'user_address': user_address,
            'contracts': [{
                'contract_address': contract_address,
                'price': price
            }]
        })
    
    with open(file_path, 'wb') as file:
        file.write(orjson.dumps(wallet_data, option=orjson.OPT_INDENT_2))


async def _check_user():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wallet_dir = os.path.normpath(os.path.join(current_dir, '../../data'))
    file_path = os.path.join(wallet_dir, 'transactions.json')

    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())
    
    list_user = []
    for user in wallet_data['users']:
        for contract in user['contracts']:
            contract_address = contract['contract_address']
            price = contract['price']
            list_user.append((user, contract_address, price))
    
    return list_user
    
    
async def _check_current_price(contract_address):
    response = requests.get(f'https://kolective-backend.vercel.app/api/token/data')
    if response.status_code == 200:
        data = response.json()
        for token in data['tokens']:
            if token['addressToken'] == contract_address:
                return token['priceChange24H']
        

async def _check_profit_followed_kol(user_address):
    response = requests.get("https://kolective-backend.vercel.app/api/kol/followed/{user_address}")
    if response.status_code == 200:
        data = response.json()
        return data['followedKOL']['kol']['avgProfitD']

async def _get_risk_profile(user_address):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wallet_dir = os.path.normpath(os.path.join(current_dir, '../../data'))
    file_path = os.path.join(wallet_dir, 'wallet.json')

    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan.")
        return []

    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())

    for data in wallet_data:
        if data['user_address'] == user_address:
            return data['risk_profile']
        

# Main function
async def transactions(action: str):
    while True:
        try:
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
                                
                                await asyncio.sleep(0)
                                print(f"  Contract {contract}: {allocated_amount:.2f}")
                                print("Executing transaction...")
                                
                                tx_hash = await agent.swap(
                                    user_address=user_address,
                                    token_in=await get_address_data('SONIC'),
                                    token_out=contract,
                                    amount=allocated_amount
                                )
                                print(f"Successful Buy Trade! txhash: {tx_hash}")
                                
                                await _update_tweet_transaction(user_address, tweet_id)
                                price = await query_graphql(tx_hash)
                                await _update_user_transactions(user_address, contract, price)

                    case 'sell':
                        processed_tweets = await _check_tweet_transaction(user_address)
                        
                        if processed_tweets:
                            for contract, _ in contract_allocations.items():
                                await asyncio.sleep(0)
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
        
        except Exception as e:
            print(f"Error in transactions function: {e}")
            await asyncio.sleep(1)
            
        await asyncio.sleep(5)
        

async def monitoring_agent(event: str):
    match event:
        case 'profit-monitoring':
            for user, contract_address, buy_price in await _check_user():
                current_price = await _check_current_price(contract_address)
                avg_profit = await _check_profit_followed_kol(user)
                
                if current_price >= buy_price * (1 + avg_profit / 100):
                    tx_hash = await agent.swap_sell(
                        user_address=user,
                        token_in=contract_address,
                        token_out=await get_address_data('SONIC')
                    )
                    print(f"Successful Sell Trade! txhash: {tx_hash}")
                
        case 'cut-loss-monitoring':
            for user, contract_address, buy_price in await _check_user():
                user_risk = await _get_risk_profile(user)   
                
                match user_risk:
                    case 'high':
                        current_price = await _check_current_price(contract_address)
                        if current_price <= buy_price * (1 - 0.15):
                            tx_hash = await agent.swap_sell(
                                user_address=user,
                                token_in=contract_address,
                                token_out=await get_address_data('SONIC')
                            )
                            print(f"Successful Sell Trade! txhash: {tx_hash}")
                            
                    case 'medium':
                        current_price = await _check_current_price(contract_address)
                        if current_price <= buy_price * (1 - 0.1):
                            tx_hash = await agent.swap_sell(
                                user_address=user,
                                token_in=contract_address,
                                token_out=await get_address_data('SONIC')
                            )
                            print(f"Successful Sell Trade! txhash: {tx_hash}")
                            
                    case 'low':
                        current_price = await _check_current_price(contract_address)
                        if current_price <= buy_price * (1 - 0.05):
                            tx_hash = await agent.swap_sell(
                                user_address=user,
                                token_in=contract_address,
                                token_out=await get_address_data('SONIC')
                            )
                            print(f"Successful Sell Trade! txhash: {tx_hash}")