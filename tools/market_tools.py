import yfinance as yf
import pandas as pd
import re
from langchain_core.tools import tool


def sanitize_output(text: str, max_length: int = 3000) -> str:
    """
    Sanitize tool output for LLM safety.
    
    Args:
        text: Raw output from yfinance or tool processing
        max_length: Maximum allowed characters (default 3000)
        
    Returns:
        Sanitized, attributed output string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove content between brackets that could contain code/instructions
    text = re.sub(r'<[^>]*>', '', text)  # Remove <...>
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [...]
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    # Add data source attribution
    if not text.strip().startswith("DATA SOURCE:"):
        text = f"DATA SOURCE: Yahoo Finance — {text.strip()}"
    
    return text.strip()


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get current stock price and basic trading information.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        Current price, day high/low, 52-week high/low, and volume as formatted string.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="2d")
        
        if hist.empty:
            return f"No data found for ticker {ticker}"
            
        current_price = hist['Close'].iloc[-1]
        day_high = hist['High'].iloc[-1]
        day_low = hist['Low'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        
        week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
        week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
        
        return sanitize_output(f"""{ticker.upper()} Stock Price Information:

Current Price: ${current_price:.2f}
Day High: ${day_high:.2f}
Day Low: ${day_low:.2f}
52-Week High: ${week_52_high}
52-Week Low: ${week_52_low}
Volume: {volume:,}

Source: Yahoo Finance via yfinance""")
        
    except Exception as e:
        return f"Error retrieving stock price for {ticker}: {str(e)}"


@tool
def get_stock_fundamentals(ticker: str) -> str:
    """
    Get fundamental financial metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        P/E ratio, market cap, EPS, dividend yield, sector, and industry as formatted string.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        eps = info.get('trailingEps', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        # Format market cap and dividend yield
        if market_cap != 'N/A' and market_cap:
            market_cap = f"${market_cap/1e9:.1f}B"
        if dividend_yield != 'N/A' and dividend_yield:
            dividend_yield = f"{dividend_yield*100:.2f}%"
            
        return sanitize_output(f"""{ticker.upper()} Fundamental Analysis:

P/E Ratio: {pe_ratio}
Market Cap: {market_cap}
EPS (Trailing): {eps}
Dividend Yield: {dividend_yield}
Sector: {sector}
Industry: {industry}

Source: Yahoo Finance via yfinance""")
        
    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


@tool
def get_price_history(ticker: str, period: str = "3mo") -> str:
    """
    Get historical price data and statistics for a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        period: Time period - valid options: 1mo, 3mo, 6mo, 1y, 2y, 5y
        
    Returns:
        Start price, end price, % change, average volume, max drawdown as formatted string.
    """
    try:
        valid_periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        if period not in valid_periods:
            return f"Invalid period. Use one of: {', '.join(valid_periods)}"
            
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"No historical data found for {ticker}"
            
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        avg_volume = hist['Volume'].mean()
        
        # Calculate max drawdown
        rolling_max = hist['Close'].expanding().max()
        drawdown = ((hist['Close'] - rolling_max) / rolling_max) * 100
        max_drawdown = drawdown.min()
        
        return sanitize_output(f"""{ticker.upper()} Price History ({period}):

Start Price: ${start_price:.2f}
End Price: ${end_price:.2f}
Total Return: {pct_change:.2f}%
Average Volume: {avg_volume:,.0f}
Max Drawdown: {max_drawdown:.2f}%

Source: Yahoo Finance via yfinance""")
        
    except Exception as e:
        return f"Error retrieving price history for {ticker}: {str(e)}"


@tool
def get_earnings_history(ticker: str) -> str:
    """
    Get earnings history with surprises for a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        Last 4 quarters of EPS actual vs estimate and surprise % as formatted string.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        earnings = stock.earnings_dates
        
        if earnings is None or earnings.empty:
            return f"No earnings data found for {ticker}"
            
        # Get last 4 quarters
        recent_earnings = earnings.head(4)
        result = f"{ticker.upper()} Earnings History (Last 4 Quarters):"
        
        for date, row in recent_earnings.iterrows():
            # Convert pandas Timestamp to datetime and format quarter
            quarter = f"{date.year}-Q{(date.month-1)//3 + 1}"
            actual_eps = row.get('Reported EPS', 'N/A')
            estimated_eps = row.get('EPS Estimate', 'N/A')
            surprise_pct = row.get('Surprise(%)', 'N/A')
            
            # Format surprise percentage if available
            if surprise_pct != 'N/A' and not pd.isna(surprise_pct):
                surprise_str = f"{surprise_pct:+.2f}%"
            else:
                surprise_str = "N/A"
                
            result += f"""

{quarter}: EPS Actual ${actual_eps} | EPS Estimate ${estimated_eps} | Surprise {surprise_str}"""
            
        result += """

Source: Yahoo Finance via yfinance"""
        return sanitize_output(result)
        
    except Exception as e:
        return f"Error retrieving earnings history for {ticker}: {str(e)}"


@tool
def compare_stocks(tickers: str) -> str:
    """
    Compare multiple stocks side-by-side.
    
    Args:
        tickers: Comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOGL")
        
    Returns:
        Side-by-side comparison: price, P/E ratio, market cap, 1yr return as formatted string.
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        result = "Stock Comparison:\n"
        result += f"{'Ticker':<8} {'Price':<10} {'P/E':<8} {'Market Cap':<12} {'1Y Return':<10}\n"
        result += "-" * 60 + "\n"
        
        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1y")
                
                price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                
                # Calculate 1-year return
                if not hist.empty and len(hist) > 1:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    one_yr_return = ((end_price - start_price) / start_price) * 100
                    return_str = f"{one_yr_return:+.1f}%"
                else:
                    return_str = "N/A"
                
                # Format market cap
                if market_cap != 'N/A' and market_cap:
                    market_cap = f"${market_cap/1e9:.1f}B"
                    
                result += f"{ticker:<8} ${price:<9.2f} {pe_ratio:<8} {market_cap:<12} {return_str:<10}\n"
                
            except Exception:
                result += f"{ticker:<8} Error retrieving data\n"
                
        result += "\n\nSource: Yahoo Finance via yfinance"
        return sanitize_output(result)
        
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"


@tool
def search_ticker(company_name: str) -> str:
    """
    Search for a company's ticker symbol by name.
    
    Args:
        company_name: Full or partial company name (e.g., "Apple" or "Berkshire Hathaway")
        
    Returns:
        Ticker symbol and company name as formatted string.
    """
    try:
        # Try to get ticker directly using yfinance search
        # yfinance doesn't have a direct search function, so we'll try common patterns
        search_term = company_name.strip().upper()
        
        # First try direct ticker lookup
        try:
            stock = yf.Ticker(search_term)
            info = stock.info
            if info and 'longName' in info:
                return f"Found: {search_term} - {info['longName']}\n\nSource: Yahoo Finance via yfinance"
        except:
            pass
            
        # Try some common stock mappings for well-known companies
        common_mappings = {
            'APPLE': 'AAPL',
            'MICROSOFT': 'MSFT',
            'GOOGLE': 'GOOGL',
            'ALPHABET': 'GOOGL',
            'AMAZON': 'AMZN',
            'TESLA': 'TSLA',
            'META': 'META',
            'FACEBOOK': 'META',
            'BERKSHIRE HATHAWAY': 'BRK-B',
            'NVIDIA': 'NVDA',
            'NETFLIX': 'NFLX',
            'JPMORGAN': 'JPM',
            'BANK OF AMERICA': 'BAC',
            'WALMART': 'WMT'
        }
        
        # Try exact match first
        if search_term in common_mappings:
            ticker = common_mappings[search_term]
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name_full = info.get('longName', 'Unknown Company')
            return sanitize_output(f"Found: {ticker} - {company_name_full}\n\nSource: Yahoo Finance via yfinance")
            
        # Try partial match
        for name, ticker in common_mappings.items():
            if search_term in name or name in search_term:
                stock = yf.Ticker(ticker)
                info = stock.info
                company_name_full = info.get('longName', 'Unknown Company')
                return sanitize_output(f"Found: {ticker} - {company_name_full}\n\nSource: Yahoo Finance via yfinance")
                
        return sanitize_output(f"No exact match found for '{company_name}'. Try the exact company name or ticker symbol.\n\nSource: Yahoo Finance via yfinance")
        
    except Exception as e:
        return f"Error searching for ticker: {str(e)}"
