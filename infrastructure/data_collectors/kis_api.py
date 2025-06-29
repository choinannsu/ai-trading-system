"""
Korea Investment Securities (KIS) API implementation for Korean stocks
Official REST API with full macOS compatibility
"""

import json
import jwt
import hashlib
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

import aiohttp

try:
    import websockets
except ImportError:
    websockets = None

from .base_api import TradingAPI
from .models import (
    MarketData, TickData, Order, OrderRequest, Position, Balance, 
    AccountInfo, Trade, Symbol, ExchangeType, WebSocketMessage,
    OrderSide, OrderType, OrderStatus, AssetType, TimeFrame
)
from .rate_limiter import rate_limit
from utils.logger import get_logger
from utils.exceptions import APIError, TradingError, ValidationError

logger = get_logger(__name__)


class KisAPI(TradingAPI):
    """
    Korea Investment Securities (KIS) API implementation
    
    Official REST API with OAuth2 authentication
    Full macOS compatibility with native HTTP requests
    """
    
    def __init__(self, api_key: str, secret_key: str, account_no: str = "", environment: str = "demo"):
        super().__init__(api_key, secret_key, environment)
        self.session: Optional[aiohttp.ClientSession] = None
        self.account_no = account_no
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
        # KIS specific settings
        self.app_key = api_key
        self.app_secret = secret_key
        
    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.KIS
    
    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return "https://openapi.koreainvestment.com:9443"
        return "https://openapivts.koreainvestment.com:29443"
    
    @property
    def websocket_url(self) -> str:
        if self.environment == "live":
            return "ws://ops.koreainvestment.com:21000"
        return "ws://ops.koreainvestment.com:31000"
    
    def _get_headers(self, token_required: bool = True) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "KIS-API/1.0"
        }
        
        if token_required and self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        headers["appkey"] = self.app_key
        headers["appsecret"] = self.app_secret
        
        return headers
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _ensure_token(self) -> None:
        """Ensure we have a valid access token"""
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires):
            await self._get_access_token()
    
    async def _get_access_token(self) -> None:
        """Get OAuth2 access token"""
        try:
            session = await self._get_session()
            
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            url = f"{self.base_url}/oauth2/tokenP"
            
            async with session.post(url, json=data, headers=self._get_headers(token_required=False)) as response:
                result = await response.json()
                
                if response.status != 200:
                    raise APIError(f"Token request failed: {result.get('msg1', 'Unknown error')}", api_name="kis")
                
                self.access_token = result["access_token"]
                expires_in = int(result["expires_in"])
                self.token_expires = datetime.now() + timedelta(seconds=expires_in - 60)  # Refresh 1 minute early
                
                logger.info("KIS access token obtained successfully")
                
        except Exception as e:
            logger.error(f"Failed to get KIS access token: {str(e)}")
            raise APIError(f"Authentication failed: {str(e)}", api_name="kis")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with authentication"""
        await self.rate_limiter.acquire(self.exchange_type.value, endpoint)
        await self._ensure_token()
        
        url = f"{self.base_url}/{endpoint}"
        session = await self._get_session()
        
        try:
            headers = self._get_headers()
            if 'headers' in kwargs:
                headers.update(kwargs.pop('headers'))
            
            async with session.request(method, url, headers=headers, **kwargs) as response:
                data = await response.json()
                
                if response.status >= 400:
                    error_message = data.get('msg1', f'HTTP {response.status}')
                    raise APIError(
                        f"KIS API error: {error_message}",
                        api_name="kis",
                        status_code=response.status,
                        response=str(data)
                    )
                
                # Check for API-level errors
                if data.get('rt_cd') != '0':
                    error_msg = data.get('msg1', 'Unknown API error')
                    raise APIError(f"KIS API error: {error_msg}", api_name="kis")
                
                return data
                
        except aiohttp.ClientError as e:
            raise APIError(f"HTTP request failed: {str(e)}", api_name="kis")
    
    # Market Data Methods
    @rate_limit("kis", "inquire_price")
    async def get_market_data(self, symbol: str, timeframe: str = "1m", 
                            limit: int = 100) -> List[MarketData]:
        """Get historical market data"""
        try:
            kis_symbol = self._format_symbol(symbol)
            
            # KIS uses different endpoints for different timeframes
            if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
                endpoint = "uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
                params = {
                    "FID_ETC_CLS_CODE": "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": kis_symbol,
                    "FID_INPUT_HOUR_1": self._convert_timeframe(timeframe),
                    "FID_PW_DATA_INCU_YN": "Y"
                }
            else:
                endpoint = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
                params = {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": kis_symbol,
                    "FID_INPUT_DATE_1": "",
                    "FID_INPUT_DATE_2": "",
                    "FID_PERIOD_DIV_CODE": "D"
                }
            
            data = await self._make_request("GET", endpoint, params=params)
            
            market_data = []
            for bar in data.get("output2", []):
                market_data.append(MarketData(
                    symbol=symbol,
                    exchange=self.exchange_type,
                    timestamp=datetime.strptime(bar["stck_bsop_date"], "%Y%m%d"),
                    open=Decimal(str(bar["stck_oprc"])),
                    high=Decimal(str(bar["stck_hgpr"])),
                    low=Decimal(str(bar["stck_lwpr"])),
                    close=Decimal(str(bar["stck_clpr"])),
                    volume=Decimal(str(bar["acml_vol"])),
                    timeframe=TimeFrame(timeframe)
                ))
            
            return market_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            raise APIError(f"Failed to get market data: {str(e)}", api_name="kis")
    
    @rate_limit("kis", "inquire_price")
    async def get_current_price(self, symbol: str) -> TickData:
        """Get current price for symbol"""
        try:
            kis_symbol = self._format_symbol(symbol)
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": kis_symbol
            }
            
            data = await self._make_request("GET", "uapi/domestic-stock/v1/quotations/inquire-price", params=params)
            
            output = data.get("output", {})
            
            return TickData(
                symbol=symbol,
                exchange=self.exchange_type,
                timestamp=datetime.now(),
                price=Decimal(str(output.get("stck_prpr", "0"))),
                size=Decimal("0"),
                bid=Decimal(str(output.get("stck_bidp", "0"))),
                ask=Decimal(str(output.get("stck_askp", "0"))),
                bid_size=Decimal(str(output.get("stck_bidp_rsqn", "0"))),
                ask_size=Decimal(str(output.get("stck_askp_rsqn", "0")))
            )
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            raise APIError(f"Failed to get current price: {str(e)}", api_name="kis")
    
    @rate_limit("kis", "inquire_search")
    async def get_symbols(self) -> List[Symbol]:
        """Get available trading symbols"""
        try:
            # Get KOSPI symbols
            kospi_params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": "0000"
            }
            
            kospi_data = await self._make_request("GET", "uapi/domestic-stock/v1/quotations/search-stock-info", params=kospi_params)
            
            symbols = []
            
            for symbol_info in kospi_data.get("output", []):
                symbols.append(Symbol(
                    symbol=symbol_info["mksc_shrn_iscd"],
                    name=symbol_info["hts_kor_isnm"],
                    exchange=self.exchange_type,
                    asset_type=AssetType.STOCK,
                    min_quantity=Decimal("1"),
                    quantity_increment=Decimal("1"),
                    price_increment=Decimal("1"),
                    is_tradable=True,
                    market_open="09:00",
                    market_close="15:30",
                    timezone="Asia/Seoul"
                ))
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise APIError(f"Failed to get symbols: {str(e)}", api_name="kis")
    
    # Trading Methods
    @rate_limit("kis", "order")
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a trading order"""
        try:
            kis_symbol = self._format_symbol(order_request.symbol)
            
            order_data = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "PDNO": kis_symbol,
                "ORD_DVSN": self._convert_order_type(order_request.type),
                "ORD_QTY": str(int(order_request.quantity)),
                "ORD_UNPR": str(int(order_request.price or 0))
            }
            
            # Determine endpoint based on order side
            if order_request.side == OrderSide.BUY:
                endpoint = "uapi/domestic-stock/v1/trading/order-cash"
            else:
                endpoint = "uapi/domestic-stock/v1/trading/order-cash"
                order_data["ORD_DVSN"] = "01"  # Sell order
            
            data = await self._make_request("POST", endpoint, json=order_data)
            
            return self._parse_order(data, order_request)
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise TradingError(f"Failed to place order: {str(e)}", symbol=order_request.symbol)
    
    @rate_limit("kis", "order")
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an existing order"""
        try:
            cancel_data = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "KRX_FWDG_ORD_ORGNO": order_id[:3],
                "ORGN_ODNO": order_id[3:],
                "ORD_DVSN": "00"
            }
            
            data = await self._make_request("POST", "uapi/domestic-stock/v1/trading/order-rvsecncl", json=cancel_data)
            
            return await self.get_order(order_id)
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise TradingError(f"Failed to cancel order: {str(e)}")
    
    @rate_limit("kis", "inquire")
    async def get_order(self, order_id: str) -> Order:
        """Get order details"""
        try:
            params = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "ODNO": order_id
            }
            
            data = await self._make_request("GET", "uapi/domestic-stock/v1/trading/inquire-order-detail", params=params)
            
            return self._parse_order_detail(data.get("output", {}))
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            raise APIError(f"Failed to get order: {str(e)}", api_name="kis")
    
    @rate_limit("kis", "inquire")
    async def get_orders(self, symbol: Optional[str] = None, 
                        status: Optional[str] = None) -> List[Order]:
        """Get orders list"""
        try:
            params = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "INQR_STRT_DT": (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
                "SLL_BUY_DVSN_CD": "00"
            }
            
            if symbol:
                params["PDNO"] = self._format_symbol(symbol)
            
            data = await self._make_request("GET", "uapi/domestic-stock/v1/trading/inquire-order", params=params)
            
            orders = []
            for order_data in data.get("output", []):
                order = self._parse_order_detail(order_data)
                
                if not status or order.status.value == status:
                    orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise APIError(f"Failed to get orders: {str(e)}", api_name="kis")
    
    # Account Methods
    @rate_limit("kis", "inquire_balance")
    async def get_balance(self) -> List[Balance]:
        """Get account balance"""
        try:
            params = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": ""
            }
            
            data = await self._make_request("GET", "uapi/domestic-stock/v1/trading/inquire-balance", params=params)
            
            output = data.get("output2", [{}])[0] if data.get("output2") else {}
            
            return [Balance(
                exchange=self.exchange_type,
                currency="KRW",
                total=Decimal(output.get("tot_evlu_amt", "0")),
                available=Decimal(output.get("ord_psbl_cash", "0")),
                locked=Decimal(output.get("tot_evlu_amt", "0")) - Decimal(output.get("ord_psbl_cash", "0")),
                usd_value=None,
                updated_at=datetime.now()
            )]
            
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            raise APIError(f"Failed to get balance: {str(e)}", api_name="kis")
    
    @rate_limit("kis", "inquire_balance")
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            params = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": ""
            }
            
            data = await self._make_request("GET", "uapi/domestic-stock/v1/trading/inquire-balance", params=params)
            
            positions = []
            for pos_data in data.get("output1", []):
                if int(pos_data.get("hldg_qty", "0")) > 0:
                    positions.append(Position(
                        symbol=pos_data["pdno"],
                        exchange=self.exchange_type,
                        asset_type=AssetType.STOCK,
                        quantity=Decimal(pos_data["hldg_qty"]),
                        avg_price=Decimal(pos_data["pchs_avg_pric"]),
                        market_value=Decimal(pos_data["evlu_amt"]),
                        unrealized_pnl=Decimal(pos_data["evlu_pfls_amt"]),
                        opened_at=datetime.now(),
                        updated_at=datetime.now()
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise APIError(f"Failed to get positions: {str(e)}", api_name="kis")
    
    @rate_limit("kis", "inquire_balance")
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            balances = await self.get_balance()
            
            return AccountInfo(
                exchange=self.exchange_type,
                account_id=self.account_no,
                is_active=True,
                trading_enabled=True,
                balances=balances,
                updated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise APIError(f"Failed to get account info: {str(e)}", api_name="kis")
    
    @rate_limit("kis", "inquire")
    async def get_trades(self, symbol: Optional[str] = None, 
                        limit: int = 100) -> List[Trade]:
        """Get trade history"""
        try:
            params = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": "01",
                "INQR_STRT_DT": (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
                "SLL_BUY_DVSN_CD": "00"
            }
            
            if symbol:
                params["PDNO"] = self._format_symbol(symbol)
            
            data = await self._make_request("GET", "uapi/domestic-stock/v1/trading/inquire-ccld", params=params)
            
            trades = []
            for trade_data in data.get("output", []):
                trades.append(Trade(
                    id=trade_data["odno"],
                    order_id=trade_data["odno"],
                    symbol=trade_data["pdno"],
                    exchange=self.exchange_type,
                    side=OrderSide.BUY if trade_data["sll_buy_dvsn_cd"] == "02" else OrderSide.SELL,
                    quantity=Decimal(trade_data["ccld_qty"]),
                    price=Decimal(trade_data["ccld_unpr"]),
                    timestamp=datetime.strptime(trade_data["ord_dt"], "%Y%m%d")
                ))
            
            return trades[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            raise APIError(f"Failed to get trades: {str(e)}", api_name="kis")
    
    # WebSocket Methods
    async def _authenticate_websocket(self) -> Dict[str, Any]:
        """Create WebSocket authentication message"""
        await self._ensure_token()
        
        return {
            "header": {
                "approval_key": self.access_token,
                "custtype": "P",
                "tr_type": "1",
                "content-type": "utf-8"
            }
        }
    
    async def _create_subscription_message(self, channel: str, symbol: str) -> Dict[str, Any]:
        """Create subscription message for WebSocket"""
        kis_symbol = self._format_symbol(symbol)
        
        if channel == "price":
            return {
                "header": {
                    "tr_id": "H0STCNT0",
                    "custtype": "P"
                },
                "body": {
                    "input": {
                        "tr_id": "H0STCNT0",
                        "tr_key": kis_symbol
                    }
                }
            }
        
        return {}
    
    async def _parse_websocket_message(self, message: Dict[str, Any]) -> Optional[WebSocketMessage]:
        """Parse incoming WebSocket message"""
        try:
            if "body" in message and "output" in message["body"]:
                output = message["body"]["output"]
                tr_id = message.get("header", {}).get("tr_id", "")
                
                if tr_id == "H0STCNT0":
                    return WebSocketMessage(
                        channel="price",
                        exchange=self.exchange_type,
                        timestamp=datetime.now(),
                        message_type="tick_data",
                        symbol=output.get("mksc_shrn_iscd", ""),
                        data={
                            "symbol": output.get("mksc_shrn_iscd", ""),
                            "exchange": self.exchange_type,
                            "timestamp": datetime.now(),
                            "price": Decimal(str(output.get("stck_prpr", "0"))),
                            "size": Decimal(str(output.get("cntg_vol", "0"))),
                            "bid": Decimal(str(output.get("stck_bidp", "0"))),
                            "ask": Decimal(str(output.get("stck_askp", "0")))
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {str(e)}")
            return None
    
    # Helper Methods
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for KIS API (6-digit Korean stock code)"""
        digits_only = ''.join(filter(str.isdigit, symbol))
        return digits_only.zfill(6)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to KIS format"""
        mapping = {
            "1m": "1",
            "5m": "5", 
            "15m": "15",
            "30m": "30",
            "1h": "60"
        }
        return mapping.get(timeframe, "1")
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to KIS format"""
        mapping = {
            OrderType.MARKET: "01",
            OrderType.LIMIT: "00",
            OrderType.STOP: "03",
            OrderType.STOP_LIMIT: "05"
        }
        return mapping.get(order_type, "00")
    
    def _parse_order(self, order_data: Dict[str, Any], order_request: OrderRequest) -> Order:
        """Parse KIS order response"""
        output = order_data.get("output", {})
        
        return Order(
            id=output.get("odno", ""),
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            exchange=self.exchange_type,
            side=order_request.side,
            type=order_request.type,
            status=OrderStatus.PENDING,
            quantity=order_request.quantity,
            filled_quantity=Decimal("0"),
            remaining_quantity=order_request.quantity,
            price=order_request.price,
            created_at=datetime.now(),
            time_in_force=order_request.time_in_force
        )
    
    def _parse_order_detail(self, order_data: Dict[str, Any]) -> Order:
        """Parse detailed order information"""
        return Order(
            id=order_data.get("odno", ""),
            symbol=order_data.get("pdno", ""),
            exchange=self.exchange_type,
            side=OrderSide.BUY if order_data.get("sll_buy_dvsn_cd") == "02" else OrderSide.SELL,
            type=self._parse_order_type(order_data.get("ord_dvsn", "00")),
            status=self._parse_order_status(order_data.get("ord_gno_brno", "")),
            quantity=Decimal(order_data.get("ord_qty", "0")),
            filled_quantity=Decimal(order_data.get("tot_ccld_qty", "0")),
            remaining_quantity=Decimal(order_data.get("ord_qty", "0")) - Decimal(order_data.get("tot_ccld_qty", "0")),
            price=Decimal(order_data.get("ord_unpr", "0")) if order_data.get("ord_unpr", "0") != "0" else None,
            avg_fill_price=Decimal(order_data.get("avg_prvs", "0")) if order_data.get("avg_prvs", "0") != "0" else None,
            created_at=datetime.strptime(order_data.get("ord_dt", "19700101"), "%Y%m%d"),
            time_in_force="GTC"
        )
    
    def _parse_order_type(self, kis_type: str) -> OrderType:
        """Parse KIS order type"""
        mapping = {
            "00": OrderType.LIMIT,
            "01": OrderType.MARKET,
            "03": OrderType.STOP,
            "05": OrderType.STOP_LIMIT
        }
        return mapping.get(kis_type, OrderType.LIMIT)
    
    def _parse_order_status(self, status_code: str) -> OrderStatus:
        """Parse KIS order status"""
        if not status_code or status_code == "":
            return OrderStatus.PENDING
        elif "체결" in status_code:
            return OrderStatus.FILLED
        elif "취소" in status_code:
            return OrderStatus.CANCELLED
        else:
            return OrderStatus.PENDING
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.disconnect()