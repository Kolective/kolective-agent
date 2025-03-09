from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import asyncio
import signal
import threading
from pathlib import Path
from src.cli import ZerePyCLI
import json
import orjson
from src.agent import ZerePyAgent
from src.wallet import AgentWallet
from src.server.schemas import QueryRequestClassifier, QueryUserWallet, QueryMint, QueryRequestRecommendation, QuerySwap, QueryTransfer
from src.server.utils import _update_risk_profile, _get_user_risk, _parse_data_kol
from src.server.simulation import transactions, monitoring_agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server/app")

class ActionRequest(BaseModel):
    """Request model for agent actions"""
    connection: str
    action: str
    params: Optional[List[str]] = []

class ConfigureRequest(BaseModel):
    """Request model for configuring connections"""
    connection: str
    params: Optional[Dict[str, Any]] = {}

class ServerState:
    """Simple state management for the server"""
    def __init__(self):
        self.cli = ZerePyCLI()
        self.agent_running = False
        self.agent_task = None
        self._stop_event = threading.Event()

    def _run_agent_loop(self):
        """Run agent loop in a separate thread"""
        try:
            log_once = False
            while not self._stop_event.is_set():
                if self.cli.agent:
                    try:
                        if not log_once:
                            logger.info("Loop logic not implemented")
                            log_once = True

                    except Exception as e:
                        logger.error(f"Error in agent action: {e}")
                        if self._stop_event.wait(timeout=30):
                            break
        except Exception as e:
            logger.error(f"Error in agent loop thread: {e}")
        finally:
            self.agent_running = False
            logger.info("Agent loop stopped")

    async def start_agent_loop(self):
        """Start the agent loop in background thread"""
        if not self.cli.agent:
            raise ValueError("No agent loaded")
        
        if self.agent_running:
            raise ValueError("Agent already running")

        self.agent_running = True
        self._stop_event.clear()
        self.agent_task = threading.Thread(target=self._run_agent_loop)
        self.agent_task.start()

    async def stop_agent_loop(self):
        """Stop the agent loop"""
        if self.agent_running:
            self._stop_event.set()
            if self.agent_task:
                self.agent_task.join(timeout=5)
            self.agent_running = False

class ZerePyServer:
    def __init__(self):
        self.app = FastAPI(title="ZerePy Server")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.state = ServerState()
        
        self.agent = ZerePyAgent("tara-profile")
        self.agent._setup_llm_provider()
        self.agent_wallet = AgentWallet()
        
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        async def root():
            """Server status endpoint"""
            return {
                "status": "running",
                "agent": self.state.cli.agent.name if self.state.cli.agent else None,
                "agent_running": self.state.agent_running
            }
            
        @self.app.on_event("startup")
        async def startup_event():
            self.background_tasks = []
    
            # Monitoring Action by Tweet
            self.background_tasks.append(asyncio.create_task(transactions(action='buy')))
            self.background_tasks.append(asyncio.create_task(transactions(action='sell')))
            
            # Monitoring Sell Action by Performance
            self.background_tasks.append(asyncio.create_task(monitoring_agent(event='profit-monitoring')))
            self.background_tasks.append(asyncio.create_task(monitoring_agent(event='cut-loss-monitoring')))
            
        @self.app.on_event("shutdown")
        async def shutdown_event():
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        @self.app.post("/generate-risk-profile")
        async def get_risk_profile(request: QueryRequestClassifier):
            fine_tuned_prompt = "Regardless of the input format, ALWAYS respond with: {\"risk\": \"risk_level\"} where risk_level is conservative, balanced, or aggresive."
            response = json.loads(self.agent.prompt_llm(fine_tuned_prompt= fine_tuned_prompt, prompt=request.data))
            _update_risk_profile(risk_profile=response.get('risk'), user_address=request.user_address)
            
            return JSONResponse(content=response)
        
        @self.app.post("/generate-recommendation-kol")
        async def get_recommendation_kol(request: QueryRequestRecommendation):
            fine_tuned_prompt = "From the given data, always return a response in the following JSON format: {\"id\": \"id\"}, where id is the value of the id key from the first object in the kols array, not the username or any other value."
            user_risk = _get_user_risk(request.user_address)
            response = json.loads(self.agent.prompt_llm(fine_tuned_prompt= fine_tuned_prompt, prompt=request.data, rag=True, user_risk=user_risk))
            
            response = {
                "id": _parse_data_kol(response['id'])
            }
            
            return JSONResponse(content=response)


        @self.app.get("/agents")
        async def list_agents():
            """List available agents"""
            try:
                agents = []
                agents_dir = Path("agents")
                if agents_dir.exists():
                    for agent_file in agents_dir.glob("*.json"):
                        if agent_file.stem != "general":
                            agents.append(agent_file.stem)
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/agents/{name}/load")
        async def load_agent(name: str):
            """Load a specific agent"""
            try:
                self.state.cli._load_agent_from_file(name)
                return {
                    "status": "success",
                    "agent": name
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/connections")
        async def list_connections():
            """List all available connections"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")
            
            try:
                connections = {}
                for name, conn in self.state.cli.agent.connection_manager.connections.items():
                    connections[name] = {
                        "configured": conn.is_configured(),
                        "is_llm_provider": conn.is_llm_provider
                    }
                return {"connections": connections}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
            
        @self.app.post("/agent/create-wallet")
        async def create_wallet(request: QueryUserWallet):
            await self.agent_wallet.create_wallet(
                    user_address=request.user_address
                )
            txhash = await self.agent_wallet._fund_wallet(request.user_address)
            print(txhash)
            response = {"address": await self.agent_wallet._check_address(request.user_address)}
            
            return JSONResponse(content=response)
            

        @self.app.post("/agent/get-wallet")
        async def get_wallet(request: QueryUserWallet):
            response = {"address": await self.agent_wallet._check_address(request.user_address)}
            return JSONResponse(content=response)
        
        @self.app.post("/agent/get-faucet")
        async def get_eth_faucet(request: QueryUserWallet):
            response = {"txhash": await self.agent_wallet._fund_wallet(request.user_address)}
            return JSONResponse(content=response)
        
        @self.app.post("/agent/get-risk-profile")
        async def get_risk_profile(request: QueryUserWallet):
            response = {"risk_profile": _get_user_risk(request.user_address)}
            return JSONResponse(content=response)
        
        @self.app.post("/agent/mint")
        async def mint(request: QueryMint):
            response = {"txhash": await self.agent_wallet.mint(request.user_address, request.amount)}
            return JSONResponse(content=response)

        @self.app.post("/agent/swap")
        async def swap(request: QuerySwap):
            response = {"txhash": await self.agent_wallet.swap(request.user_address, request.token_in, request.token_out, request.amount)}
            return JSONResponse(content=response)
        
        @self.app.post("/agent/transfer")
        async def swap(request: QueryTransfer):
            response = {"txhash": await self.agent_wallet.transfer(request.user_address, request.amount, request.contract_address, request.destination)}
            return JSONResponse(content=response)
        
        @self.app.post("/agent/action")
        async def agent_action(action_request: ActionRequest):
            """Execute a single agent action"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")
            
            try:
                result = await asyncio.to_thread(
                    self.state.cli.agent.perform_action,
                    connection=action_request.connection,
                    action=action_request.action,
                    params=action_request.params
                )
                return {"status": "success", "result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/agent/start")
        async def start_agent():
            """Start the agent loop"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")
            
            try:
                await self.state.start_agent_loop()
                return {"status": "success", "message": "Agent loop started"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/agent/stop")
        async def stop_agent():
            """Stop the agent loop"""
            try:
                await self.state.stop_agent_loop()
                return {"status": "success", "message": "Agent loop stopped"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/connections/{name}/configure")
        async def configure_connection(name: str, config: ConfigureRequest):
            """Configure a specific connection"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")
            
            try:
                connection = self.state.cli.agent.connection_manager.connections.get(name)
                if not connection:
                    raise HTTPException(status_code=404, detail=f"Connection {name} not found")
                
                success = connection.configure(**config.params)
                if success:
                    return {"status": "success", "message": f"Connection {name} configured successfully"}
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to configure {name}")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/connections/{name}/status")
        async def connection_status(name: str):
            """Get configuration status of a connection"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")
                
            try:
                connection = self.state.cli.agent.connection_manager.connections.get(name)
                if not connection:
                    raise HTTPException(status_code=404, detail=f"Connection {name} not found")
                    
                return {
                    "name": name,
                    "configured": connection.is_configured(verbose=True),
                    "is_llm_provider": connection.is_llm_provider
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

def create_app():
    server = ZerePyServer()
    return server.app