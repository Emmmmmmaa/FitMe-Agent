# import asyncio
# import os

# import azure.identity
# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.messages import TextMessage
# from autogen_core import CancellationToken
# from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
# from dotenv import load_dotenv

# # Setup the client to use either Azure OpenAI or GitHub Models
# load_dotenv(override=True)
# API_HOST = os.getenv("API_HOST", "github")
# if API_HOST == "github":
#     client = OpenAIChatCompletionClient(model=os.getenv("GITHUB_MODEL", "gpt-4o"), api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
# elif API_HOST == "azure":
#     token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
#     client = AzureOpenAIChatCompletionClient(model=os.environ["AZURE_OPENAI_CHAT_MODEL"], api_version=os.environ["AZURE_OPENAI_VERSION"], azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_ad_token_provider=token_provider)


# agent = AssistantAgent(
#     "clothing_analyst",
#     model_client=client,
#     system_message="""You are a professional clothing image analyst with expertise in fashion and style. Your task is to analyze clothing images and provide detailed descriptions.""",
# )


# async def main() -> None:
#     # You can use either local file path or image URL
#     image_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"  # Or stable image URL
#     # image_source = "https://i.ibb.co/fzFrgxpH/1067e98499a9.jpg"
    
#     # Construct the message based on the image source
#     message = f"""Please analyze this clothing image and provide a detailed description:
#     Image source: {image_source}"""
    
#     response = await agent.on_messages(
#         [TextMessage(content=message, source="user")],
#         cancellation_token=CancellationToken(),
#     )
#     print(response.chat_message.content)


# if __name__ == "__main__":
#     asyncio.run(main())