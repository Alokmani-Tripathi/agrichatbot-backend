import os
import requests
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a farming location.
    Use when user asks about rain, temperature, humidity, or weather.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={location}&appid={api_key}&units=metric"
        )
        r    = requests.get(url, timeout=5)
        data = r.json()
        if r.status_code != 200:
            return f"Could not fetch weather for '{location}'."
        weather  = data["weather"][0]["description"]
        temp     = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind     = data["wind"]["speed"]
        return (
            f"Weather in {location}:\n"
            f"Condition : {weather}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity  : {humidity}%\n"
            f"Wind speed: {wind} m/s"
        )
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def get_mandi_price(crop: str, state: str = "Maharashtra") -> str:
    """
    Get today's mandi/market price for a crop.
    Use when user asks about crop prices, mandi rates, or market prices.
    """
    # Layer 1 — data.gov.in
    try:
        api_key = "579b464db66ec23d955859aa819f8a25"
        url = (
            f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            f"?api-key={api_key}&format=json&limit=5"
            f"&filters[commodity]={crop}&filters[state]={state}"
        )
        r       = requests.get(url, timeout=8)
        data    = r.json()
        records = data.get("records", [])

        if records:
            result = f"Mandi prices for {crop} in {state}:\n"
            for rec in records[:3]:
                result += (
                    f"  Market: {rec.get('market','N/A')} | "
                    f"Min: ₹{rec.get('min_price','N/A')} | "
                    f"Max: ₹{rec.get('max_price','N/A')} | "
                    f"Modal: ₹{rec.get('modal_price','N/A')}\n"
                )
            return result

    except Exception:
        pass  # move to Layer 2

    # Layer 2 — Tavily web search fallback
    try:
        results = tavily_client.search(
            query=f"{crop} mandi price today {state} India",
            max_results=3,
            search_depth="basic",
        )
        output = f"Mandi prices for {crop} in {state} (from web search):\n\n"
        for r in results.get("results", []):
            output += (
                f"• {r['title']}\n"
                f"  {r['content'][:200]}...\n"
                f"  🔗 Source: {r['url']}\n\n"
            )
        return output if output else "No price data found."

    except Exception:
        pass  # move to Layer 3

    # Layer 3 — Manual fallback
    return (
        f"Unable to fetch live mandi price for {crop} right now.\n"
        f"Please check: https://agmarknet.gov.in for latest prices."
    )


@tool
def web_search(query: str) -> str:
    """
    Search the web for agricultural information not in the knowledge base.
    Use for recent news, pest outbreaks, government schemes, or anything not in PDFs.
    """
    try:
        results = tavily_client.search(
            query=f"agriculture farming India {query}",
            max_results=3,
            search_depth="basic",
        )
        output = ""
        for r in results.get("results", []):
            output += (
                f"• {r['title']}\n"
                f"  {r['content'][:200]}...\n"
                f"  🔗 Source: {r['url']}\n\n"
            )
        return output if output else "No relevant results found."
    except Exception as e:
        return f"Error during web search: {str(e)}"


agri_tools = [get_weather, get_mandi_price, web_search]