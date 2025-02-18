import asyncio
from pyppeteer import launch

html_template = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.2.3/mermaid.min.js"></script>
</head>
<body>
    <div class="mermaid">
            flowchart TD
                A[Start]
                
                %% process_uploaded_image Flow
                A --> C[Call process_uploaded_image(file)]
                C --> D[Invoke vision_client.vision_to_text(file)]
                D --> E{Success?}
                E -- Yes --> F[Return car_details]
                E -- No --> G[Log error and raise HTTPException]
                
                %% handle_sql_mode Flow
                A --> H[Call handle_sql_mode(user_prompt)]
                H --> I[Invoke sql_agent.chat_agent_with_sql(user_prompt)]
                I --> J[Return SQL assistant_response]
                
                %% handle_normal_chat_mode Flow
                A --> K[Call handle_normal_chat_mode(user_prompt, conversation_history, car_details)]
                K --> L[Combine conversation_history, user_prompt, & car_details]
                L --> M[Generate prompt using prompt_template.text_propt_user]
                M --> N[Invoke text_generation_client.generate_text(prompt)]
                N --> O[Return chat assistant_response]
                
                %% ReAct Agent Loop Flow
                A --> P[Call react_agent(user_prompt, conversation_history, car_details)]
                P --> Q[Initialize messages list]
                Q --> R[Append system prompt (react_system_prompt)]
                R --> S[Append conversation history (if available)]
                S --> T[Append car details (if available)]
                T --> U[Append user's message]
                U --> V[Start For Loop (max 6 iterations)]
                
                V --> W[Call text_generation_client.generate_text(prompt, chat_history=messages)]
                W --> X[Append assistant_reply to messages]
                X --> Y{Reply contains "Answer:"?}
                Y -- Yes --> Z[Extract final answer and return it]
                Y -- No --> AA{Reply contains "Action:"?}
                AA -- Yes --> AB[Parse tool_name and tool_input]
                AB --> AC{Tool Name}
                AC -- "handle_sql_mode" --> AD[Call handle_sql_mode(tool_input)]
                AC -- "handle_normal_chat_mode" --> AE[Call handle_normal_chat_mode(tool_input, conversation_history, car_details)]
                AC -- "process_uploaded_image" --> AF[Mock process_uploaded_image(tool_input)]
                AD & AE & AF --> AG[Append "Observation: <result>" to messages]
                AG --> V
                AA -- No --> V
                V --> AH[Loop ended without "Answer:"]
                AH --> AI[Return error message "I'm sorry, but I couldn't find a final answer."]
    </div>
    <script>
        mermaid.initialize({ startOnLoad: true });
    </script>
</body>
</html>
"""

async def render_mermaid():
    browser = await launch()
    page = await browser.newPage()
    await page.setContent(html_template)
    await page.waitForSelector(".mermaid")  # انتظر حتى يتم تحميل Mermaid
    await page.screenshot({"path": "workflow.png"})
    await browser.close()
    print("تم حفظ المخطط بنجاح!")

asyncio.run(render_mermaid())
