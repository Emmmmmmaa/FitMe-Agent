import autogen
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from typing import Dict, List, Optional, Tuple, Union
import json
import os
import pandas as pd
from dotenv import load_dotenv
import azure.identity
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

load_dotenv(override=True)

class FashionAgent:
    def __init__(self, input_data: Union[pd.DataFrame, str, List[Dict]]):
        """初始化FashionAgent
        
        参数:
            input_data: 可以是以下类型之一:
                - pd.DataFrame: 直接的DataFrame数据
                - str: CSV文件路径或者衣物文本描述
                - List[Dict]: 已解析的服装列表
        """
        # 配置 LLM
        API_HOST = os.getenv("API_HOST", "github")
        print(f"使用API_HOST: {API_HOST}")
        if API_HOST == "github":
            self.client = OpenAIChatCompletionClient(
                model=os.getenv("GITHUB_MODEL", "gpt-4o"), 
                api_key=os.environ["GITHUB_TOKEN"],
                base_url="https://models.inference.ai.azure.com"
            )
        elif API_HOST == "azure":
            token_provider = azure.identity.get_bearer_token_provider(
                azure.identity.DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAIChatCompletionClient(
                model=os.environ["AZURE_OPENAI_CHAT_MODEL"],
                api_version=os.environ["AZURE_OPENAI_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_ad_token_provider=token_provider
            )
        
        # 处理输入数据
        self.input_type = self._determine_input_type(input_data)
        self.clothing_data = self._process_input_data(input_data)
        
        # 创建智能体
        self.fashion_agent = AssistantAgent(
            "fashion_expert",
            model_client=self.client,
            system_message="""You are a professional fashion recommendation system. Your task is to recommend suitable clothing combinations based on user input.

            1. Data Constraints and Categories:
                - Each item has a 'type' field that indicates its category:
                   * '上衣': Upper body clothing (shirts, t-shirts, jackets, etc.)
                   * '下装': Lower body clothing (pants, skirts, etc.)
                   * '配饰': Accessories (hats, shoes, bags, etc.)
                   * '连衣裙/裤': Dresses and jumpsuits
                
                - RECOMMENDATION RULES:
                   * EITHER select exactly one '上衣' and one '下装' item
                   * OR select one '连衣裙/裤' item
                   * '配饰' items are optional additional suggestions
                
                - CRITICAL: Trust the 'type' field in the data. Do not reclassify items.
                - NEVER recommend a '配饰' item as clothing
                - Cannot fabricate or modify item attributes

            2. Style Understanding (CRITICAL):
                - PRIORITY: Always analyze the 'title' field to determine actual style, even if 'style' field exists
                - Understand style keywords in item titles rather than relying on the predefined style field
                - Common style types and their appropriate items:
                   * 'Sporty/Athletic' (运动风): T-shirts, shorts, sports pants, sneakers, athletic wear, hoodies
                   * 'Casual' (休闲): Jeans, t-shirts, casual shirts, comfortable everyday wear
                   * 'Formal' (正式): Suits, dress shirts, formal dresses, blazers, ties
                   * 'Elegant' (优雅): Well-tailored items, quality fabrics, refined designs
                   * 'Bohemian' (波西米亚): Flowy fabrics, earthy tones, loose-fitting garments
                   * 'Vintage' (复古): Retro designs, classic patterns, historical references
                   * 'Minimalist' (极简): Clean lines, neutral colors, unembellished designs
                
                - STYLE MATCHING GUIDELINES:
                   * For 'Sporty': Prioritize comfortable, athletic-looking pieces (T-shirts, sneakers, shorts, sports pants)
                   * For 'Casual': Prioritize everyday comfortable wear
                   * For 'Formal': Prioritize structured, elegant pieces
                
                - Never recommend items that clearly contradict the requested style (e.g., evening gowns for sporty style)
                - If no items match the requested style well, choose the closest alternatives and note the limitation

            3. Color matching: 
                - Similar main colors > Complementary > Other contrasts (Priority: High)
                - IMPORTANT: For sporty style, prefer bright, vibrant, or contrasting colors

            4. Temperature Rules (ABSOLUTE HIGHEST PRIORITY - MUST CHECK FIRST):
                STRICT TEMPERATURE VALIDATION:
                - For temperature > 25°C:
                    FORBIDDEN MATERIALS (AUTO-REJECT):
                    * Wool (羊毛)
                    * Fleece (摇粒绒/绒)
                    * Thick materials (厚)
                    * Warm materials (保暖)
                    REQUIRED: Light, breathable materials only
                
                - For temperature < 15°C:
                    FORBIDDEN:
                    * Short sleeves
                    * Shorts
                    * Thin materials
                    REQUIRED: Warm materials only
                
                - For temperature 15-25°C:
                    ALLOWED: Medium thickness materials
                    CHECK: Balanced warmth between top and bottom

                CRITICAL: If temperature is provided, you MUST check material keywords in item titles BEFORE making any recommendation.
                REJECT any item containing forbidden materials for the given temperature.

            5. Mood-based Color Matching (Secondary Priority):
                - Happy: Bright, vibrant colors
                - Calm: Soft, neutral colors
                - Sad: Dark, low-saturation colors

            6. VERIFICATION PROCESS (MANDATORY):
               Before finalizing selection, verify:
               - If recommending upper+lower: Both items are properly categorized
               - If recommending a dress: It is properly categorized as '连衣裙/裤'
               - No accessories are being recommended as clothing
               - STYLE CHECK: Final selection must truly match the requested style in function and appearance
               - If you cannot find proper items, state this clearly

            Output Format (JSON):
            {
                "top": {
                    "title": "Top name",
                    "image_url": "Image URL",
                    "type": "上衣"
                },
                "bottom": {
                    "title": "Bottom name",
                    "image_url": "Image URL",
                    "type": "下装"
                },
                "reason": "Detailed explanation of the recommendation"
            }
            
            OR if recommending a dress:
            
            {
                "dress": {
                    "title": "Dress name",
                    "image_url": "Image URL",
                    "type": "连衣裙/裤"
                },
                "reason": "Detailed explanation of the recommendation"
            }

            CRITICAL REQUIREMENTS:
            1. ALL OUTPUT MUST BE IN ENGLISH, regardless of input language
            2. Before making any recommendation, you MUST verify temperature compatibility first
            3. If any temperature rule is violated, REJECT the item immediately and choose another
            4. Reason must explain why the combination suits user's needs
            5. Include color, style, and material details in descriptions
            6. ALWAYS use the 'type' field to determine item category"""
        )
    
    def _determine_input_type(self, input_data):
        """确定输入数据类型"""
        if isinstance(input_data, pd.DataFrame):
            return "dataframe"
        elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            return "clothing_list"
        elif isinstance(input_data, str):
            # 检查是否是文件路径
            if os.path.exists(input_data) and input_data.lower().endswith('.csv'):
                return "csv_path"
            else:
                return "text_description"
        else:
            raise ValueError("不支持的输入类型。必须是DataFrame、字典列表、CSV文件路径或衣物文本描述。")
    
    def _process_input_data(self, input_data):
        """处理不同类型的输入数据"""
        if self.input_type == "dataframe":
            print(f"直接使用DataFrame数据，包含{len(input_data)}条记录")
            return input_data
        
        elif self.input_type == "csv_path":
            # 检查文件是否存在
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"CSV文件不存在: {input_data}")
            
            # 读取CSV文件
            data = pd.read_csv(input_data)
            print(f"成功加载CSV数据，共 {len(data)} 条记录")
            print("数据预览:")
            print(data.head())
            return data
        
        elif self.input_type == "clothing_list":
            # 已经是格式化好的服装列表
            print(f"使用提供的服装列表，包含{len(input_data)}件服装")
            return pd.DataFrame(input_data)
        
        elif self.input_type == "text_description":
            # 直接使用文本描述
            print("使用文本描述数据")
            # 创建一个只有一行的DataFrame，包含原始文本
            return pd.DataFrame([{
                "title": input_data,
                "type": "text_description",
                "is_text_description": True
            }])
    
    async def process_request(self, style_preference: str, 
                             temperature: Optional[float] = None, 
                             mood: Optional[str] = None) -> Dict:
        """处理服装推荐请求"""
        
        # 检查是否是文本描述
        if self.input_type == "text_description":
            # 直接使用原始文本进行提示
            text_description = self.clothing_data.iloc[0]["title"]
            
            # 构建文本描述的提示
            initial_prompt = f"""
            Please help me recommend clothing combinations:
            
            1. Clothing description:
            {text_description}
            
            2. User preferences:
            - Target style: {style_preference}
            - Current temperature: {temperature if temperature else 'Not specified'}
            - Current mood: {mood if mood else 'Not specified'}
            
            3. Processing instructions:
            - This is a text description, please first identify each piece of clothing and its attributes (type, color, material, etc.)
            - Then recommend suitable combinations based on the identified items
            - Must choose either one top and one bottom, or one dress/jumpsuit
            - Accessories like hats, shoes, bags cannot be recommended as main clothing items
            
            4. Style guidance:
            - IMPORTANT: Analyze the clothing descriptions to determine their actual style
            - For "sporty/athletic" style: Prioritize T-shirts, sports pants, shorts, sneakers, etc.
            - For "casual" style: Prioritize jeans, casual shirts, everyday comfortable wear
            - For "formal" style: Prioritize structured garments, elegant designs
            - Ensure the recommended items truly match the requested style in function and appearance
            
            5. Temperature rules (if temperature is provided):
            - Temperature > 25°C: Only recommend light, breathable materials, avoid wool, fleece and other warm materials
            - Temperature < 15°C: Only recommend warm materials, avoid short sleeves, shorts and other thin clothes
            - Temperature 15-25°C: Recommend materials of medium thickness
            """
        else:
            # 获取服装数据 (CSV或DataFrame方式)
            clothing_data = self.clothing_data.to_dict('records')
            
            # 构建标准提示
            initial_prompt = f"""
            Please help me recommend clothing combinations:
            
            1. Available clothing data:
            {json.dumps(clothing_data, ensure_ascii=False, indent=2)}
            
            2. User preferences:
            - Target style: {style_preference}
            - Current temperature: {temperature if temperature else 'Not specified'}
            - Current mood: {mood if mood else 'Not specified'}
            
            3. Data source: {'Imported from Taobao purchase history CSV' if self.input_type in ['csv_path', 'dataframe'] else 'Imported from list'}
            
            4. Recommendation rules:
            - Must select one top and one bottom combination, or one dress
            - Accessories like hats cannot be recommended as tops
            - Strictly follow the type field classification in the items
            - For items with type 'Unknown', please determine the category based on the title
            
            5. Style guidance (IMPORTANT):
            - Analyze the 'title' field to determine actual style, even if 'style' field exists
            - For "sporty/athletic" style: Prioritize T-shirts, sports pants, shorts, sneakers, etc.
            - For "casual" style: Prioritize jeans, casual shirts, everyday comfortable wear
            - For "formal" style: Prioritize structured garments, elegant designs
            - Ensure the recommended items truly match the requested style in function and appearance
            - Do NOT rely solely on the 'style' field in the data
            """
        
        # 发送消息并获取响应
        response = await self.fashion_agent.on_messages(
            [TextMessage(content=initial_prompt, source="user")],
            cancellation_token=CancellationToken(),
        )
        
        return response.chat_message.content

async def main():
    try:
        # 获取输入模式
        input_mode = input("请选择输入模式 (1: CSV文件, 2: 文本描述文件): ")
        
        if input_mode == "1":
            # CSV模式
            csv_path = input("请输入CSV文件路径 (默认: data/new_processed_taobao_purchases.csv): ")
            if not csv_path:
                csv_path = "data/new_processed_taobao_purchases.csv"
            
            # 创建FashionAgent实例，传入CSV文件路径
            agent = FashionAgent(csv_path)
        else:
            # 文本描述文件模式
            txt_path = "data/upload_process_txt.txt"
            with open(txt_path, 'r', encoding='utf-8') as file:
                text_description = file.read()
        
            
            # 创建FashionAgent实例，传入文本描述
            agent = FashionAgent(text_description)
        
        # 获取用户偏好
        style_preference = input("请输入你想要的风格: ")
        temperature = input("请输入当前温度(可选，直接回车跳过): ")
        temperature = float(temperature) if temperature else None
        mood = input("请输入当前心情(可选，直接回车跳过): ")
        
        # 处理推荐请求
        result = await agent.process_request(
            style_preference=style_preference,
            temperature=temperature,
            mood=mood
        )
        
        # 打印结果
        print("\n推荐结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 