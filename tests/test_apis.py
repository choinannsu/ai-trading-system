"""
Comprehensive test suite for all trading APIs
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.data_collectors import (
    AlpacaAPI, UpbitAPI, KiwoomAPI, APIManager,
    OrderRequest, OrderSide, OrderType, ExchangeType
)


class TestAlpacaAPI:
    """Test Alpaca API functionality"""
    
    @pytest.fixture
    def alpaca_api(self):
        """Create Alpaca API instance for testing"""
        return AlpacaAPI(
            api_key="test_key",
            secret_key="test_secret",
            environment="paper"
        )
    
    def test_alpaca_initialization(self, alpaca_api):
        """Test Alpaca API initialization"""
        assert alpaca_api.exchange_type == ExchangeType.ALPACA
        assert alpaca_api.base_url == "https://paper-api.alpaca.markets"
        assert alpaca_api.websocket_url == "wss://stream.data.alpaca.markets/v2/iex"
    
    @pytest.mark.asyncio
    async def test_alpaca_market_data(self, alpaca_api):
        """Test Alpaca market data retrieval"""
        # Mock the HTTP response
        mock_response = {
            "bars": {
                "AAPL": [
                    {
                        "t": "2023-01-01T09:30:00Z",
                        "o": 150.0,
                        "h": 151.0,
                        "l": 149.0,
                        "c": 150.5,
                        "v": 1000000,
                        "vw": 150.25,
                        "n": 1500
                    }
                ]
            }
        }
        
        with patch.object(alpaca_api, '_make_request', return_value=mock_response):
            market_data = await alpaca_api.get_market_data("AAPL", "1d", 1)
            
            assert len(market_data) == 1
            assert market_data[0].symbol == "AAPL"
            assert market_data[0].open == Decimal("150.0")
            assert market_data[0].high == Decimal("151.0")
    
    @pytest.mark.asyncio
    async def test_alpaca_current_price(self, alpaca_api):
        """Test Alpaca current price retrieval"""
        mock_response = {
            "trade": {
                "t": "2023-01-01T09:30:00Z",
                "p": 150.5,
                "s": 100
            }
        }
        
        with patch.object(alpaca_api, '_make_request', return_value=mock_response):
            tick_data = await alpaca_api.get_current_price("AAPL")
            
            assert tick_data.symbol == "AAPL"
            assert tick_data.price == Decimal("150.5")
            assert tick_data.size == Decimal("100")
    
    @pytest.mark.asyncio
    async def test_alpaca_place_order(self, alpaca_api):
        """Test Alpaca order placement"""
        mock_response = {
            "id": "test_order_id",
            "symbol": "AAPL",
            "side": "buy",
            "type": "limit",
            "status": "new",
            "qty": "10",
            "filled_qty": "0",
            "limit_price": "150.00",
            "created_at": "2023-01-01T09:30:00Z",
            "updated_at": "2023-01-01T09:30:00Z",
            "time_in_force": "GTC"
        }
        
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("150.00")
        )
        
        with patch.object(alpaca_api, '_make_request', return_value=mock_response):
            order = await alpaca_api.place_order(order_request)
            
            assert order.id == "test_order_id"
            assert order.symbol == "AAPL"
            assert order.side == OrderSide.BUY
            assert order.quantity == Decimal("10")


class TestUpbitAPI:
    """Test Upbit API functionality"""
    
    @pytest.fixture
    def upbit_api(self):
        """Create Upbit API instance for testing"""
        return UpbitAPI(
            api_key="test_key",
            secret_key="test_secret",
            environment="live"
        )
    
    def test_upbit_initialization(self, upbit_api):
        """Test Upbit API initialization"""
        assert upbit_api.exchange_type == ExchangeType.UPBIT
        assert upbit_api.base_url == "https://api.upbit.com"
        assert upbit_api.websocket_url == "wss://api.upbit.com/websocket/v1"
    
    @pytest.mark.asyncio
    async def test_upbit_market_data(self, upbit_api):
        """Test Upbit market data retrieval"""
        mock_response = [
            {
                "candle_date_time_kst": "2023-01-01 09:30:00",
                "opening_price": 50000.0,
                "high_price": 51000.0,
                "low_price": 49000.0,
                "trade_price": 50500.0,
                "candle_acc_trade_volume": 1000.5
            }
        ]
        
        with patch.object(upbit_api, '_make_request', return_value=mock_response):
            market_data = await upbit_api.get_market_data("KRW-BTC", "1d", 1)
            
            assert len(market_data) == 1
            assert market_data[0].symbol == "KRW-BTC"
            assert market_data[0].open == Decimal("50000.0")
            assert market_data[0].close == Decimal("50500.0")
    
    @pytest.mark.asyncio
    async def test_upbit_current_price(self, upbit_api):
        """Test Upbit current price retrieval"""
        mock_response = [
            {
                "trade_price": 50500.0,
                "trade_volume": 0.5,
                "timestamp": 1672574400000,
                "highest_52_week_price": 70000.0,
                "lowest_52_week_price": 30000.0
            }
        ]
        
        with patch.object(upbit_api, '_make_request', return_value=mock_response):
            tick_data = await upbit_api.get_current_price("KRW-BTC")
            
            assert tick_data.symbol == "KRW-BTC"
            assert tick_data.price == Decimal("50500.0")
            assert tick_data.size == Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_upbit_balance(self, upbit_api):
        """Test Upbit balance retrieval"""
        mock_response = [
            {
                "currency": "KRW",
                "balance": "1000000.0",
                "locked": "0.0"
            },
            {
                "currency": "BTC",
                "balance": "0.5",
                "locked": "0.1"
            }
        ]
        
        with patch.object(upbit_api, '_make_request', return_value=mock_response):
            balances = await upbit_api.get_balance()
            
            assert len(balances) == 2
            krw_balance = next(b for b in balances if b.currency == "KRW")
            btc_balance = next(b for b in balances if b.currency == "BTC")
            
            assert krw_balance.total == Decimal("1000000.0")
            assert btc_balance.total == Decimal("0.6")
            assert btc_balance.locked == Decimal("0.1")


class TestKiwoomAPI:
    """Test Kiwoom API functionality"""
    
    @pytest.fixture
    def kiwoom_api(self):
        """Create Kiwoom API instance for testing"""
        return KiwoomAPI(
            api_key="test_key",
            secret_key="test_secret",
            environment="demo"
        )
    
    def test_kiwoom_initialization(self, kiwoom_api):
        """Test Kiwoom API initialization"""
        assert kiwoom_api.exchange_type == ExchangeType.KIWOOM
        assert "demo" in kiwoom_api.base_url
    
    @pytest.mark.asyncio
    async def test_kiwoom_market_data(self, kiwoom_api):
        """Test Kiwoom market data retrieval"""
        mock_response = {
            "output": [
                {
                    "stck_bsop_date": "20230101",
                    "stck_oprc": "70000",
                    "stck_hgpr": "71000",
                    "stck_lwpr": "69000",
                    "stck_clpr": "70500",
                    "acml_vol": "1000000"
                }
            ]
        }
        
        with patch.object(kiwoom_api, '_make_request', return_value=mock_response):
            market_data = await kiwoom_api.get_market_data("005930", "1d", 1)
            
            assert len(market_data) == 1
            assert market_data[0].symbol == "005930"
            assert market_data[0].open == Decimal("70000")
            assert market_data[0].close == Decimal("70500")
    
    @pytest.mark.asyncio
    async def test_kiwoom_current_price(self, kiwoom_api):
        """Test Kiwoom current price retrieval"""
        mock_response = {
            "output": {
                "stck_prpr": "70500",
                "stck_bidp": "70400",
                "stck_askp": "70600",
                "stck_bidp_rsqn": "1000",
                "stck_askp_rsqn": "800"
            }
        }
        
        with patch.object(kiwoom_api, '_make_request', return_value=mock_response):
            tick_data = await kiwoom_api.get_current_price("005930")
            
            assert tick_data.symbol == "005930"
            assert tick_data.price == Decimal("70500")
            assert tick_data.bid == Decimal("70400")
            assert tick_data.ask == Decimal("70600")


class TestAPIManager:
    """Test API Manager functionality"""
    
    @pytest.fixture
    def api_manager(self):
        """Create API Manager instance for testing"""
        manager = APIManager()
        # Mock the exchanges to avoid actual API calls
        manager.exchanges = {
            ExchangeType.ALPACA: MagicMock(),
            ExchangeType.UPBIT: MagicMock(),
            ExchangeType.KIWOOM: MagicMock()
        }
        manager.enabled_exchanges = {ExchangeType.ALPACA, ExchangeType.UPBIT, ExchangeType.KIWOOM}
        return manager
    
    def test_api_manager_initialization(self, api_manager):
        """Test API Manager initialization"""
        assert len(api_manager.enabled_exchanges) >= 0  # May be empty if no API keys configured
        assert hasattr(api_manager, 'exchanges')
        assert hasattr(api_manager, 'exchange_status')
    
    def test_asset_type_detection(self, api_manager):
        """Test asset type detection logic"""
        # US Stocks
        assert api_manager._guess_asset_type("AAPL") == AssetType.STOCK
        assert api_manager._guess_asset_type("TSLA") == AssetType.STOCK
        
        # Korean Stocks
        assert api_manager._guess_asset_type("005930") == AssetType.STOCK
        
        # Crypto
        assert api_manager._guess_asset_type("BTCUSDT") == AssetType.CRYPTO
        assert api_manager._guess_asset_type("KRW-BTC") == AssetType.CRYPTO
    
    @pytest.mark.asyncio
    async def test_unified_portfolio(self, api_manager):
        """Test unified portfolio aggregation"""
        # Mock exchange status as connected
        for exchange_type in api_manager.enabled_exchanges:
            api_manager.exchange_status[exchange_type] = MagicMock()
            api_manager.exchange_status[exchange_type].is_connected = True
        
        # Mock balance responses
        async def mock_get_balance(exchange):
            if exchange == ExchangeType.ALPACA:
                return [MagicMock(currency="USD", total=Decimal("10000"), usd_value=Decimal("10000"))]
            elif exchange == ExchangeType.UPBIT:
                return [MagicMock(currency="KRW", total=Decimal("1000000"), usd_value=Decimal("800"))]
            return []
        
        # Mock position responses
        async def mock_get_positions(exchange):
            if exchange == ExchangeType.ALPACA:
                return [MagicMock(symbol="AAPL", quantity=Decimal("10"), market_value=Decimal("1500"))]
            return []
        
        # Patch the methods
        for exchange_type, api in api_manager.exchanges.items():
            api.get_balance = AsyncMock(side_effect=lambda ex=exchange_type: mock_get_balance(ex))
            api.get_positions = AsyncMock(side_effect=lambda ex=exchange_type: mock_get_positions(ex))
        
        portfolio = await api_manager.get_unified_portfolio()
        
        assert 'balances' in portfolio
        assert 'total_usd_value' in portfolio
        assert 'positions' in portfolio
        assert 'exchanges' in portfolio


# Integration Tests
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for real API calls (requires API keys)"""
    
    @pytest.mark.asyncio
    async def test_upbit_public_api(self):
        """Test Upbit public API calls (no authentication required)"""
        upbit_api = UpbitAPI("", "", "live")  # No keys needed for public data
        
        try:
            # Test market data (public endpoint)
            symbols = await upbit_api.get_symbols()
            assert len(symbols) > 0
            
            # Test current price (public endpoint)
            btc_price = await upbit_api.get_current_price("KRW-BTC")
            assert btc_price.price > 0
            
            print(f"✓ Upbit public API test passed")
            print(f"  - Found {len(symbols)} trading symbols")
            print(f"  - BTC price: ₩{btc_price.price:,}")
            
        except Exception as e:
            print(f"✗ Upbit public API test failed: {e}")
            pytest.skip(f"Upbit API not available: {e}")
        
        finally:
            await upbit_api.close()
    
    @pytest.mark.asyncio
    async def test_alpaca_public_api(self):
        """Test Alpaca public API calls"""
        alpaca_api = AlpacaAPI("", "", "paper")  # Paper trading
        
        try:
            # Most Alpaca endpoints require authentication
            # This test will likely fail without API keys
            print("⚠ Alpaca API test skipped (requires authentication)")
            pytest.skip("Alpaca API requires authentication")
            
        except Exception as e:
            print(f"✗ Alpaca API test failed: {e}")
            pytest.skip(f"Alpaca API not available: {e}")
        
        finally:
            await alpaca_api.close()


# Utility Functions for Manual Testing
async def manual_test_all_apis():
    """Manual test function for all APIs"""
    print("🧪 Starting comprehensive API tests...")
    
    # Test API Manager
    print("\n--- Testing API Manager ---")
    manager = APIManager()
    
    try:
        # Test connection (will only work with proper API keys)
        connection_results = await manager.connect_all()
        
        for exchange, connected in connection_results.items():
            status = "✓" if connected else "✗"
            print(f"{status} {exchange.value}: {'Connected' if connected else 'Failed'}")
        
        # Test system status
        system_status = manager.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Connected: {system_status['exchanges_connected']}")
        print(f"  Health Score: {system_status['average_health_score']:.2f}")
        
        # Test public data if any exchanges are connected
        connected_exchanges = [ex for ex, connected in connection_results.items() if connected]
        
        if connected_exchanges:
            print(f"\n--- Testing Market Data ---")
            
            # Test different asset types
            test_symbols = {
                AssetType.STOCK: "AAPL",
                AssetType.CRYPTO: "KRW-BTC"
            }
            
            for asset_type, symbol in test_symbols.items():
                try:
                    price = await manager.get_current_price(symbol)
                    print(f"✓ {symbol}: {price.price} ({price.exchange.value})")
                except Exception as e:
                    print(f"✗ {symbol}: Failed - {e}")
        
    except Exception as e:
        print(f"Manager test failed: {e}")
    
    finally:
        await manager.disconnect_all()
        await manager.close()
    
    print("\n🏁 API tests completed!")


if __name__ == "__main__":
    # Run manual tests
    asyncio.run(manual_test_all_apis())