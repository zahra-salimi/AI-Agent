#  Standard Python Libraries
import os
import json
import re
from typing import List, Optional, TypedDict

#  Environment & Data Validation
from dotenv import load_dotenv
from pydantic import BaseModel

# LangChain & LangGraph Core
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

# Messaging System
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# Document Processing & Retrieval
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- RTL Display Support ---
import re
import arabic_reshaper
from bidi.algorithm import get_display


# --- RTL printing helpers ---

_ARABIC_RE = re.compile(r'[\u0600-\u06FF]')

def _rtl(text: str) -> str:
    """Formats RTL (Arabic/Persian) text for correct terminal display.

       Uses `arabic_reshaper` for letter shaping and `bidi` for reordering
       characters only when RTL characters are detected in the input string.

       Args:
           text (str): The input string to format.

       Returns:
           str: The formatted string ready for proper RTL display.
    """
    if not isinstance(text, str):
        text = str(text)
    if _ARABIC_RE.search(text):
        return get_display(arabic_reshaper.reshape(text))
    return text

def rprint(*args, **kwargs):
    print(*(_rtl(a) for a in args), **kwargs)


# --------------- Initial settings ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

llm_name = "gpt-4o"
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=llm_name)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, chunk_size=100)


# --------------- Data preparation ----------------
def setup_retriever(file_path: str):
    """
        Loads, cleans, and prepares JSON product data into a FAISS-based retriever for semantic search.

        This function performs the following steps:
          1. Loads product descriptions from a JSON file using `JSONLoader`.
          2. Cleans HTML tags from text content.
          3. Adds product metadata (e.g., name and id) to each document.
          4. Splits documents into manageable text chunks for better embedding performance.
          5. Builds a FAISS vector store using pre-initialized OpenAI embeddings.
          6. Returns a retriever object for efficient similarity-based querying.

        Args:
            file_path (str): Path to the JSON file containing product data.
                             Each item should include at least "id", "name", and "description" fields.

        Returns:
            BaseRetriever: A FAISS retriever instance ready for use in a LangChain pipeline.

        Prints:
            - Number of original and split documents.
            - Confirmation message upon successful FAISS vector store creation.
    """
    loader = JSONLoader(file_path=file_path, jq_schema='.[]', content_key="description")
    documents = loader.load()
    for doc in documents:
        doc.page_content = re.sub(r'<[^>]+>', '', doc.page_content)

    with open(file_path, 'r', encoding='utf-8') as f:
        products_data = json.load(f)
    for doc, product in zip(documents, products_data):
        doc.metadata["name"] = product["name"]
        doc.metadata["id"] = product["id"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)

    rprint(f"تعداد اسناد اولیه: {len(documents)}")
    rprint(f"تعداد اسناد پس از تقسیم: {len(split_documents)}")

    vector_store = FAISS.from_documents(split_documents, embeddings)
    rprint("--- Vector Store با موفقیت ایجاد شد. ---")
    return vector_store.as_retriever()


product_retriever = setup_retriever('./data/products.json')




# ---------------  Definition of the situation and models ----------------------------
class BusinessInfo(BaseModel):
    """
        Represents structured business information extracted from user input or dialogue context.

        This model is used to store and manage the core attributes of a business that are
        necessary for personalized analysis and product recommendations. Each field is optional
        because the data may be collected progressively during a multi-turn conversation.

        Attributes:
            business_type (Optional[str]):
                The type or category of the business — for example, "clothing store",
                "restaurant", "digital marketing agency", etc.

            customer_type (Optional[str]):
                Indicates whether the business primarily serves B2B (business-to-business)
                or B2C (business-to-consumer) customers.

            geo_location (Optional[str]):
                The geographic area in which the business operates, such as a city,
                province, or region (e.g., "Tehran", "Isfahan").

            online_presence (Optional[str]):
                Describes the online visibility of the business — for example,
                "website", "Instagram page", "online store", or "none".

        Example:
            >>> info = BusinessInfo(
            ...     business_type="online shoe store",
            ...     customer_type="B2C",
            ...     geo_location="Iran",
            ...     online_presence="website"
            ... )
    """
    business_type: Optional[str] = None

    customer_type: Optional[str] = None

    geo_location: Optional[str] = None

    online_presence: Optional[str] = None



class AgentState(TypedDict):
    """
       Represents the full conversational and reasoning state of the business consultant agent.

       This structure defines all key elements that persist between nodes in the LangGraph workflow.
       It keeps track of the ongoing dialogue, extracted business information, analytical results,
       and generated product recommendations — allowing the agent to make context-aware decisions
       at each step of the conversation.

       Attributes:
           messages (List[BaseMessage]):
               The full conversation history between the user and the assistant,
               including both `HumanMessage` and `SystemMessage` objects.

           business_info (BusinessInfo):
               Structured information about the user's business, collected progressively
               through conversation (e.g., business type, customer type, location, online presence).

           analysis (Optional[str]):
               The generated business analysis or summary based on the collected information.
               May include insights, challenges, or suggestions.

           product_recommendations (Optional[str]):
               A text summary containing recommended products or services relevant to the user's business.

           next_node (str):
               The name of the next node to execute in the LangGraph workflow.
               Used for routing conversation flow (e.g., `"generate_analysis"`, `"product_search"`, `"end"`).

       Example:
           >>> state = AgentState(
           ...     messages=[HumanMessage(content="I run an online shoe store.")],
           ...     business_info=BusinessInfo(business_type="shoe store", customer_type="B2C"),
           ...     analysis=None,
           ...     product_recommendations=None,
           ...     next_node="generate_analysis"
           ... )
    """
    messages: List[BaseMessage]
    business_info: BusinessInfo
    analysis: Optional[str]
    product_recommendations: Optional[str]
    next_node: str


# --------------- (Nodes) --------------------------------
def master_router_node(state: AgentState) -> dict:
    """
        Acts as the central decision-making node of the business consultant agent.

        This function analyzes the latest user message in the conversation and decides
        what the next step in the dialogue should be. It is responsible for distinguishing
        between a *new conversation* (first user message) and an *ongoing conversation*
        (follow-up answers), extracting or updating business information, and routing
        to the appropriate node.

        **Behavior:**

        1. **New Conversation**
           - Detects the user’s intent:
             - `"consultation"` if the user is describing their business.
             - `"product_search"` if the user is asking directly for a product.
           - If it’s a consultation, extracts basic business info from the message.
           - If the business type is missing or too general, it asks a clarifying question.
           - Then, following a fixed priority (`business_type → customer_type → geo_location → online_presence`),
             it asks the next missing question or moves to the analysis step when all information is complete.

        2. **Ongoing Conversation**
           - Determines whether the latest user reply is an answer to the previous question
             or a new product-related query.
           - If it’s a new query, routes to the `product_search` node.
           - Otherwise, it updates existing business information with the new answer.
           - Rechecks for missing or unclear fields and asks the next required question in order.
           - Once all required fields are filled, it routes to `generate_analysis`.

        **Validation Rules:**
           - The model avoids guessing: only explicitly stated information is extracted.
           - `customer_type` is considered valid **only if** clear indicators (e.g., "B2B", "B2C", "consumer", "corporate") appear in the text.
           - Business descriptions that are too general (like “I started a page”) trigger a clarification question.

        Args:
            state (AgentState): The current conversation state containing message history,
                                business info, and next node metadata.

        Returns:
            dict: A dictionary update for the agent’s state containing:
                  - Updated `business_info` (if applicable)
                  - Appended `messages` (when a new question is asked)
                  - The `next_node` key indicating the next step in the workflow.
    """

    last_message_content = state["messages"][-1].content
    is_new_conversation = len(state["messages"]) <= 1



    # Consider customer_type valid only if explicit B2B/B2C or equivalent terms are found in the text
    EXPLICIT_CT_PAT = re.compile(
        r'\bB2B\b|\bB2C\b|مصرف[\s‌]*کننده|سازمانی|شرکتی',
        re.IGNORECASE
    )

    def enforce_explicit_customer_type(info: BusinessInfo, source_text: str) -> BusinessInfo:
        if info.customer_type and not EXPLICIT_CT_PAT.search(source_text or ""):
            info.customer_type = None
        return info

    def is_general_sentence(text: str) -> bool:
        if not text:
            return True
        check_prompt = f"""
        جمله زیر را بررسی کن و فقط یکی از دو واژه را بنویس (هیچ توضیح دیگری ننویس):
        - general
        - specific
        جمله: "{text}"
        """
        resp = model.invoke(check_prompt).content.strip().lower()
        return "general" in resp

    priority = ["business_type", "customer_type", "geo_location", "online_presence"]
    questions_map = {
        "business_type": "نوع کسب‌وکار شما چیه؟ چه محصول یا خدماتی ارائه می‌دین؟",
        "customer_type": "مشتریان شما B2C (مصرف‌کننده نهایی) هستند یا B2B (سازمانی/شرکتی)؟",
        "geo_location": "کسب‌وکار شما در چه موقعیت جغرافیایی فعالیت می‌کند؟",
        "online_presence": "آیا وب‌سایت یا صفحه فروش فعال دارید؟"
    }

    # --- Case 1: Brand new conversation ---
    if is_new_conversation:
        # Intent detection: consultation vs. product search
        routing_prompt = f"""
        Analyze the user's intent from their first message.
        - If they are describing their business for consultation (e.g., "I have a shop..."), respond with "consultation".
        - If they are directly asking for a product (e.g., "Do you have SEO courses?"), respond with "product_search".

        User message: "{last_message_content}"
        Intent:
        """
        response = model.invoke(routing_prompt)
        if "product_search" in response.content.lower():
            return {"next_node": "product_search"}


        parser = model.with_structured_output(BusinessInfo)
        try:
            extracted_info = parser.invoke(
                f"""فقط اگر کاربر در متن زیر «صراحتاً» چیزی گفته همان را استخراج کن.
اگر چیزی صریح نگفته هر کدام از فیلدها را خالی بگذار (حدس نزن).
متن: '{last_message_content}'"""
            )
        except Exception:
            extracted_info = BusinessInfo()


        if (not extracted_info.business_type) or is_general_sentence(extracted_info.business_type) or is_general_sentence(last_message_content):
            return {
                "business_info": extracted_info,
                "messages": state["messages"] + [SystemMessage(content=questions_map["business_type"])],
                "next_node": "end"
            }


        extracted_info = enforce_explicit_customer_type(extracted_info, last_message_content)


        for slot in priority:
            if not getattr(extracted_info, slot, None):
                return {
                    "business_info": extracted_info,
                    "messages": state["messages"] + [SystemMessage(content=questions_map[slot])],
                    "next_node": "end"
                }


        return {"business_info": extracted_info, "next_node": "generate_analysis"}

    # --- Case 2: Ongoing conversation ---
    else:
        last_question = state["messages"][-2].content if len(state["messages"]) >= 2 else ""

        contextual_routing_prompt = f"""
        You are a business consultant bot. Your last message to the user was a question.
        Analyze the user's latest reply in the context of your question.

        Your question was: "{last_question}"
        The user's reply is: "{last_message_content}"

        Determine the user's intent:
        - If the user's reply is a direct answer to your question (e.g., answering "location?" with "Tehran"), respond with "consultation_answer".
        - If the user is ignoring your question and asking for a new product or service, respond with "product_search".

        Intent:
        """
        response = model.invoke(contextual_routing_prompt)
        if "product_search" in response.content.lower():
            return {"next_node": "product_search"}


        extraction_prompt = (
            f"""با توجه به سوال '{last_question}', فقط اگر کاربر «صراحتاً» پاسخی داده همان را استخراج کن.
اگر مشخص نیست، فیلد را خالی بگذار (حدس نزن).
پاسخ کاربر: '{last_message_content}'"""
        )
        parser = model.with_structured_output(BusinessInfo)
        try:
            extracted_info = parser.invoke(extraction_prompt)
        except Exception:
            extracted_info = BusinessInfo()


        current_info = state.get("business_info", BusinessInfo())
        updated_info_dict = current_info.model_dump()
        if extracted_info:
            for key, value in extracted_info.model_dump().items():
                if value is not None:
                    updated_info_dict[key] = value
        updated_info = BusinessInfo(**updated_info_dict)


        all_user_text = " ".join(m.content for m in state["messages"] if isinstance(m, HumanMessage))
        updated_info = enforce_explicit_customer_type(updated_info, all_user_text)


        if (not updated_info.business_type) or is_general_sentence(updated_info.business_type):
            return {
                "business_info": updated_info,
                "messages": state["messages"] + [SystemMessage(content="لطفاً نوع کسب‌وکار خود را دقیق‌تر بفرمایید (مثلاً: فروش لباس زنانه، طراحی سایت، آموزش سئو و ...).")],
                "next_node": "end"
            }


        for slot in priority:
            if not getattr(updated_info, slot, None):
                return {
                    "business_info": updated_info,
                    "messages": state["messages"] + [SystemMessage(content=questions_map[slot])],
                    "next_node": "end"
                }


        return {"business_info": updated_info, "next_node": "generate_analysis"}



def product_search_node(state: AgentState) -> dict:
    """
        Performs a semantic product search using the user's latest message.

        Sends the last user query to the `product_retriever`, retrieves matching
        products, and returns an updated message list containing either the
        recommended items or a polite 'no results found' message.

        Args:
            state (AgentState): Current conversation state containing user messages.

        Returns:
            dict: Updated state with a new SystemMessage listing product results.
    """

    last_message = state["messages"][-1].content
    retrieved_docs = product_retriever.invoke(last_message)
    if not retrieved_docs:
        result_message = "متاسفانه محصولی مرتبط با درخواست شما پیدا نشد."
    else:
        unique_products = {doc.metadata["id"]: f"- **{doc.metadata['name']}**: {doc.page_content}" for doc in
                           retrieved_docs}
        result_message = "بر اساس درخواست شما، محصولات زیر پیشنهاد می‌شوند:\n\n" + "\n".join(unique_products.values())
    return {"messages": state["messages"] + [SystemMessage(content=result_message)]}


def generate_analysis_node(state: AgentState) -> dict:
    """
      Generates a short business analysis based on collected information.

      Converts the current `BusinessInfo` in state to JSON, builds a natural-language
      prompt, and uses the language model to produce a concise, actionable analysis.

      Args:
          state (AgentState): Current conversation state containing `business_info`.

      Returns:
          dict: {"analysis": <generated_text>} — the generated analysis text only.
    """
    info_json = state["business_info"].model_dump_json(indent=2)
    prompt = f"یک تحلیل کوتاه و کاربردی برای کسب‌وکار با اطلاعات زیر ارائه بده: {info_json}"
    analysis_result = model.invoke(prompt).content
    rprint("--- تحلیل اولیه ایجاد شد. ---")
    return {"analysis": analysis_result}


def recommend_products_node(state: AgentState) -> dict:
    """Generates targeted product recommendations by creating a focused semantic query.

       This node is responsible for the final step of the consultation flow: suggesting
       relevant products based on the user's profile and the generated business analysis.
       It follows a sophisticated three-step process to ensure high relevance and avoid
       the common problem of "query contamination" where the full analysis text might
       confuse the retriever.

       **Process:**
       1.  **Extract Core Need Keywords:** Instead of using the entire analysis text,
           it first uses an LLM to analyze the user's specific answer about their
           online presence. The LLM's task is to distill the user's situation into a
           short, powerful set of keywords (e.g., "website design, getting started" for a new
           business, or "SEO consulting, business growth" for an established one).

       2.  **Build a Focused Query:** It constructs a clean and highly targeted query
           for the retriever. This query combines the user's basic business info
           (type, customer) with the precise keywords extracted in the previous step.
           This prevents irrelevant details from the analysis from misleading the semantic search.

       3.  **Execute Search and Build Response:** It invokes the `product_retriever` with
           the focused query to find the most relevant product documents. The results are
           then formatted and combined with the full analysis text to create a single,
           comprehensive final message for the user.

       Args:
           state (AgentState): The current state of the graph, which must contain
               the user's `business_info` and the `analysis` text.

       Returns:
           dict: An update for the agent's state, containing the `product_recommendations`
                 text and appending the final combined `SystemMessage` (analysis +
                 recommendations) to the `messages` list.
    """

    info = state['business_info']
    analysis_text = state['analysis']

    # --- Step 1: Extracting Core User Need Keywords ---
    keyword_extraction_prompt = f"""
    یک مشتری با مشخصات زیر، در پاسخ به سوال "آیا حضور آنلاین دارید؟" چنین جوابی داده است:
    - نوع کسب‌وکار: '{info.business_type}'
    - نوع مشتریان: '{info.customer_type}'
    - پاسخ کاربر در مورد حضور آنلاین: "{info.online_presence}"

    بر اساس پاسخ کاربر، نیاز اصلی او را در قالب چند کلمه کلیدی کوتاه و اصلی خلاصه کن.

    ۱. اگر پاسخ به معنی "نداشتن" است، کلمات کلیدی باید شامل این موارد باشد: "طراحی سایت"، "راه اندازی فروشگاه آنلاین"، "مشاوره ورود به دنیای دیجیتال".
    ۲. اگر پاسخ به معنی "داشتن" است اما با ابهام یا اشاره به ضعف (مانند 'سایت داریم ولی فعال نیست')، کلمات کلیدی باید شامل این موارد باشد: "بهینه سازی سایت"، "مشاوره سئو"، "تولید محتوا برای سایت"، "بازطراحی سایت".
    ۳. اگر پاسخ یک "بله" قاطع است (مانند 'بله سایت دارم')، کلمات کلیدی باید شامل این موارد باشد: "مشاوره تخصصی سئو"، "رشد کسب و کار آنلاین"، "تبلیغات گوگل ادز"، "مدیریت شبکه های اجتماعی".

    کلمات کلیدی اصلی برای این کاربر (با کاما جدا کن):
    """

    generated_keywords = model.invoke(keyword_extraction_prompt).content.strip()

    # --- Step 2: Building the Final Query ---
    query = f"محصولات مناسب برای یک {info.business_type} برای مشتریان {info.customer_type} که نیازمند این موارد است: {generated_keywords}"

    # print(f"\n--- کوئری ساخته شده برای Retriever:\n{query}\n---")

    # --- Step 3: Execute Search and Build the Final Response ---
    retrieved_docs = product_retriever.invoke(query)

    recommendation_text = ""
    if not retrieved_docs:
        recommendation_text = "متاسفانه محصولی که دقیقاً با شرایط شما منطبق باشد پیدا نشد. اما می‌توانید لیست کامل محصولات ما را بررسی کنید."
    else:
        unique_products = {doc.metadata["id"]: f"- **{doc.metadata['name']}**: {doc.page_content}" for doc in
                           retrieved_docs}
        product_list_str = "\n".join(unique_products.values())
        recommendation_text = f"با توجه به تحلیل انجام شده، محصولات و پکیج‌های زیر می‌توانند برای شما بسیار مفید باشند:\n\n{product_list_str}"

    analysis_result = state.get("analysis", "تحلیل کسب‌وکار شما آماده است.")
    final_response = f"تحلیل اولیه کسب‌وکار شما:\n\n{analysis_result}\n\n---\n\n{recommendation_text}"

    return {
        "product_recommendations": recommendation_text,
        "messages": state["messages"] + [SystemMessage(content=final_response)]
    }
# --------------- decide_next_node -----------------------
def decide_next_node(state: AgentState) -> str:
    """
       Determines which node should run next in the workflow.

       Simply returns the value of `next_node` from the current agent state,
       which is used by LangGraph to control conversation flow.

       Args:
           state (AgentState): Current agent state containing the `next_node` key.

       Returns:
           str: The name of the next node to execute.
    """
    return state["next_node"]

#----------------------------------------------------------
# Build and connect the LangGraph workflow.
# Defines all nodes, entry point, and transition rules between steps.
# The graph routes user messages through intent detection, analysis, and product recommendation.
builder = StateGraph(AgentState)
builder.add_node("master_router", master_router_node)
builder.add_node("product_search", product_search_node)
builder.add_node("generate_analysis", generate_analysis_node)
builder.add_node("recommend_products", recommend_products_node)

builder.set_entry_point("master_router")
builder.add_conditional_edges("master_router", decide_next_node,
                              {"product_search": "product_search", "generate_analysis": "generate_analysis",
                               "end": END})
builder.add_edge("generate_analysis", "recommend_products")
builder.add_edge("product_search", END)
builder.add_edge("recommend_products", END)

graph = builder.compile()

# --------------- Run the business consultant chatbot  --------------------------------------
# Run the business consultant chatbot in interactive mode.
# Initializes the conversation state and continuously processes user input
# through the LangGraph workflow until the user exits.
if __name__ == "__main__":
    initial_state = {"messages": [], "business_info": BusinessInfo(), "next_node": ""}
    rprint("سلام! من مشاور کسب‌وکار شما هستم...")
    while True:
        user_input = input("شما: ")
        if user_input.lower() in ["خروج", "exit","q"]: break
        initial_state["messages"].append(HumanMessage(content=user_input))
        final_state = graph.invoke(initial_state)
        if final_state.get("messages") and len(final_state["messages"]) > len(initial_state["messages"]):
            rprint(f"ربات: {final_state['messages'][-1].content}")
        initial_state = final_state