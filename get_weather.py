from openai import OpenAI
import json
import os
import requests
from typing import Dict, Union

client = OpenAI()

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for the given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. San Francisco, United States",
                    },
                },
                "required": ["location"],
            },
        },
    },
]


def get_weather(location: str) -> Dict[str, Union[str, float]]:
    open_weather_map_api_key = os.getenv("OPEN_WEATHER_MAP_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?q={location}"
        f"&appid={open_weather_map_api_key}"
    )
    resp = requests.get(url)
    data = resp.json()
    print(f"OpenWeatherMap data: {data}")

    # Extract the temperature in Kelvin
    kelvin_temp = data["main"]["temp"]

    # Convert Kelvin to Celsius
    celsius_temp = kelvin_temp - 273.15

    return {
        "location": location,
        "temperature": celsius_temp,
        "city": data["name"],
        "country": data["sys"]["country"],
        "coordinates": data["coord"],
        "description": data["weather"][0]["description"],
        "pressure": data["main"]["pressure"],
        "humidity": data["main"]["humidity"],
        "sea level": data["main"]["sea_level"],
        "ground level": data["main"]["grnd_level"],
        "visibility": data["visibility"],
        "wind speed": data["wind"]["speed"],
        "wind degrees": data["wind"]["deg"],
        "clouds": data["clouds"],
        "calculation epoch timestamp": data["dt"],
    }


# Create a running list of messages we will add to over time
messages = []

print("Welcome to the Weather AI Assistant!")
print(
    "Please enter a question with city and country for the location "
    "you want to ask about the weather, e.g., "
    '"What\'s the weather like in San Francisco, United States?".'
)
print('Type "exit" to quit.')

is_running = True
while is_running is True:
    user_input = input("You: ")
    content = user_input.lower()

    if content == "exit":
        print("\nGoodbye!")
        is_running = False
    else:
        messages.append({"role": "user", "content": content})

        # 2. Prompt the model with tools defined
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )

        # Save function call messages for subsequent requests
        for choice in response.choices:
            if choice.message.tool_calls:
                print("Initial response JSON for the prompt with tools defined:")
                print(response.to_json())

                messages.append(choice.message.to_dict())
                for item in choice.message.tool_calls:
                    if item.type == "function":
                        if item.function.name == "get_weather":
                            # 3. Execute the function logic for get_weather
                            weather = get_weather(
                                json.loads(item.function.arguments)["location"]
                            )
                            print(f"Weather results: {weather}")

                            # 4. Provide function call results to the model
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": item.id,
                                    "name": item.function.name,
                                    "content": json.dumps(weather),
                                }
                            )

        print("\nFinal input debug log:")
        print(messages)

        response = client.chat.completions.create(model="gpt-5", messages=messages)

        # 5. The model should be able to give a response!
        print("\nFinal output debug log:")
        print(response.to_json())

        print("\nFinal input:")
        for message in messages:
            if message["content"]:
                print(message["content"])

        print("\nFinal output:")
        # Print output and save chat completions messages for subsequent requests
        for choice in response.choices:
            print(choice.message.content)
            messages.append(choice.message.to_dict())
        print("\n")
