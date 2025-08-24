from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.document_transformers import LongContextReorder
import deepeval_compare as DP
import cleanup as CP
import time

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


def geetreport(userquery: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        verbose=True,
        temperature=0.2,
        max_tokens=8192,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    df_logs = pd.read_csv(
        "testdata.csv"
    )

    def pretty_print_docs(docs):
        final_docs = "\n\n".join([d.page_content for i, d in enumerate(docs)])
        return final_docs

    solution_recommendation = {}

    template1 = """You are an experienced Inventory Manager responsible for overseeing product management, order management, and user management within a comprehensive inventory system. A user has submitted a query related to Inventory Management. This query could be a report of an issue, a question about product functionality, or any other inquiry related to managing inventory within the system. Please analyze the user's query and provide two distinct and plausible reasons for this to occur, considering the interconnectedness of product, order, and user management.

                User Query: {query}

                Output Format:
                Reason_1:
                Reason_2:

                Notes:
                (1) Do not use ** for the headings or descriptions; just generate the reasons.
                (2) Please do not change the reason headings. Keep the format as given above.
                (3) Consider potential issues arising from product data inaccuracies, order processing errors, user access limitations, system glitches, or integration problems between these three management areas.
                (4) Ensure each reason is specific and actionable, providing a clear direction for investigation.

    """
    question1 = userquery
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template1)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    res = llm_chain.invoke(
        {
            "query": question1,
        }
    )
    print(res["text"])
    print("------------------")
    answer = res["text"]
    if "Reason 1" not in answer:
        r1 = answer.split("Reason_1:")[1].split("Reason_2")[0].strip()
        r2 = answer.split("Reason_2:")[1].split("Reason_3")[0].strip()
    else:
        print("The answer is not in the format")
        res = llm_chain.invoke(
            {
                "query": question1,
            }
        )
        answer = res["text"]
        r1 = answer.split("Reason_1:")[1].split("Reason_2")[0].strip()
        r2 = answer.split("Reason_2:")[1].split("Reason_3")[0].strip()
    reasonlist = [r1, r2]
    count = 1
    event_list_template = """You are a highly experienced Supply Chain Architect tasked with analyzing a given issue statement and a proposed possible reason within the context of a software system designed for inventory, order, and user management. Your objective is to identify the most relevant Software System Events from the provided list that logically and semantically align with the issue and the potential cause. Consider how each module's functionality could directly or indirectly contribute to the scenario.

                User Query: {user_query}
                Possible Reason: {possible_reason}

                Software System Events:
                1) Product Management: Gain complete control over your product catalog. This module facilitates seamless inventory adjustments, allowing you to track items as they enter and leave your stock, update product information in real-time, and effortlessly remove obsolete or discontinued items.
                2) Order Management: Simplify and enhance your order fulfillment process. This module empowers users to place orders with ease and provides robust tracking capabilities, keeping you informed about the status of each order from placement to delivery.
                3) User Management: Securely administer access and roles within the system. This module allows authorized personnel to effectively manage user accounts, ensuring the right individuals have the appropriate level of access to the software's features.

                Task:
                As a Supply Chain Architect, analyze the provided Issue Statement and Possible Reason. Based on your understanding of supply chain operations and inventory manager the described software system events, identify the most relevant System Events that could logically and semantically contribute to the scenario.
                Format the output as below:
                    1) Provide the identified events in a list in separate line
                    2) Do not add any explanation or description for the identification, just list out the events.
                    3) Do not add any list format like 1), a) just prove the list of name of events.
                    4) Do not add any special characters or symbols in the output, just provide the events."""

    pc_listdf = pd.read_csv(
        "childeventlist.csv"
    )
    for reasons in reasonlist:
        event_list_prompt = PromptTemplate.from_template(event_list_template)
        event_llm_chain = LLMChain(llm=llm, prompt=event_list_prompt)
        res1 = event_llm_chain.invoke(
            {"user_query": question1, "possible_reason": reasons}
        )
        print("The event name is : ", res1["text"])
        print(res1["text"].split("\n"))
        parentevents = res1["text"].split("\n")
        df1 = pd.DataFrame(columns=["Child event", "description"])
        for parentevent in parentevents:
            print("The parent event is : ", parentevent)
            filtered_df = pc_listdf[
                pc_listdf["parentevent"] == str(parentevent).strip()
            ][["Child event", "description"]]
            df1 = pd.concat([df1, filtered_df], ignore_index=True)
        print("\n")
        print(df1)
        temp_csv_file_name = "temp" + str(count) + ".csv"
        df1.to_csv(temp_csv_file_name, index=False)
        print("--------------Stage 2 complete----------\n\n")

        loader = CSVLoader(
            file_path=".\chatappcombined\\" + temp_csv_file_name
        )
        docscsv = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
        questionemb1 = "User Query:"+question1 + "\n" + "Reason: " + reasons
        splits = text_splitter.split_documents(docscsv)
        hf = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="models/text-embedding-004",
        )
        db = faiss.FAISS.from_documents(splits, embedding=hf)
        ret = db.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25}
        )
        data = ret.get_relevant_documents(question1)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(data)
        ret_docs = pretty_print_docs(reordered_docs)
        print("The documetns ret are : \n", ret_docs)

        functionality_filter_temp = """Analyze the following Reason, User Query and Context. Using the Child event name and its description provide the best logically and sematically fits to the given Issue and Reason.
        {inputdata}
        Context: {event_con}
        
        Format the output as:
        Event_Name: Description
        
        Note:
        0) List all possible that are possible for the issue to occur.
        1) Repalce Event_name with child event name and Description with its description.
        1) Do not add any explanation or description for the identification, just list out the possible child event and its description.
        2) Do not add any list format like 1), a) just prove the list of name of event and its description.
        3) Do not add any special characters or symbols in the output, just provide the child event name and description.
        """
        event_function_pmp = PromptTemplate.from_template(functionality_filter_temp)
        llm_chain2 = LLMChain(llm=llm, prompt=event_function_pmp)
        res_function_name = llm_chain2.invoke(
            {
                "inputdata": questionemb1,
                "event_con": ret_docs,
            }
        )
        print("\n\nThe stage 3 LLM Output is  :\n\n", res_function_name["text"])
        funlist = []
        for event in res_function_name["text"].split("\n"):
            funlist.append(event.split(":")[0])
        print(funlist)
        funlist_temp = list(filter(None, funlist))
        filtered_funlist = [item for item in funlist_temp if item != ""]
        print(filtered_funlist)
        print("-------------------Stage 3 complete----------------\n")
        column_names = list(df_logs.columns)
        print("The column names are: ", column_names)
        filtered_logs_df = pd.DataFrame(columns=column_names)
        if not filtered_funlist:
            print("No logs found based on the provided function list.")
            filtered_logs_df = pd.DataFrame([{'id': None, 'Action': 'No logs found', 'details': 'No logs found', 'timestamp': None, 'username': None}])
        else:
            for funct in filtered_funlist:
                print("The function name is :", funct)
                filtered_logs_df = pd.concat(
                    [
                        filtered_logs_df,
                        df_logs[df_logs["Action"] == str(funct).strip()],
                    ],
                    ignore_index=True,
                )
        print(filtered_logs_df)
        if filtered_logs_df.empty:
            print("Filtered DataFrame is empty. Adding 'No logs found' entry.")
            filtered_logs_df = pd.DataFrame([{'id': None, 'Action': 'No logs found', 'details': 'No logs found', 'timestamp': None, 'username': None}])
        filtered_logs_csv_name = "filtered" + str(count) + ".csv"
        filtered_logs_df.to_csv(filtered_logs_csv_name, index=False)
        print("-------Stage 4 complete ----------\n")
        loadercsv = CSVLoader(
            file_path="C:\\Users\\HP\\Desktop\\chatappcombined\\"
            + filtered_logs_csv_name
        )
        docslogscsv = loadercsv.load()
        bm25_retriever = BM25Retriever.from_documents(docslogscsv)
        bm25_retriever.k = 5

        text_splitter_logs = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        splits_logs = text_splitter_logs.split_documents(docslogscsv)

        hflogs = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="models/text-embedding-004",
        )
        db_logs = faiss.FAISS.from_documents(splits_logs, embedding=hflogs)
        logs_faiss_retriever = db_logs.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25}
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, logs_faiss_retriever],
            weights=[
                0.5,
                0.5,
            ],  # You can tweak weights to prioritize one over the other
        )
        results = ensemble_retriever.get_relevant_documents(questionemb1)
        reordering = LongContextReorder()
        reordered_Logs_docs = reordering.transform_documents(results)
        ret_logs_docs = pretty_print_docs(reordered_Logs_docs)
        print("The documetns ret are : \n", ret_logs_docs)
        print("\n\n---------------llm output is ---------------\n\n")
        final_template = """As an expert Supply Chain Architect and Operations Architect, deeply analyze the following User Query, and provided logs (if any) and also the timestamp of the log occurance to perform a comprehensive Root Cause Analysis. The query might be a report of an issue, a question about product functionality, or any other inquiry related to managing inventory.
        {input_issue_reason}
        Supporting Information: {context}

        Your Task:
        Based on the provided context,timestamps and logs, perform a thorough, logical, and semantic analysis to deliver the following:

        Root Cause Analysis (RCA):
        1)Clearly identify the fundamental underlying cause(s) of the query.
        2)Explain the chain of events or factors that led to the problem.
        3)Support your analysis with evidence from the provided context and logs.
        4)If multiple contributing factors exist, clearly delineate each one and their interaction.
        (5) The query might be a report of an issue, a question about product functionality, or any other inquiry related to managing inventory

        Workaround(s):
        1)Provide immediate, temporary solutions or steps that can be taken to mitigate the impact of the issue in the short term.
        2)Explain how each workaround helps alleviate the problem.
        3)Consider practicality and ease of implementation.

        Permanent Fix(es):
        1)Outline the long-term solutions required to address the root cause(s) and prevent the issue from recurring.
        2)Provide specific and actionable steps for implementing the permanent fix(es).
        3)Consider the resources, time, and potential impact of implementing these fixes.

        Output Requirements:
        Present your analysis, workarounds, and permanent fixes in a clear, well-structured, and easily understandable manner for a non-expert consumer. Include relevant artifacts where appropriate to illustrate your points. """
        final_output_pmp = PromptTemplate.from_template(final_template)
        llm_chain_final = LLMChain(llm=llm, prompt=final_output_pmp)
        res_final = llm_chain_final.invoke(
            {
                "input_issue_reason": question1,
                "context": ret_logs_docs,
            }
        )
        print(res_final["text"])
        # score=DP.get_full_scores(res_final["text"],question1,ret_logs_docs)
        # print("THe answer relavency score is : ",score)
        solution_recommendation["r" + str(count)] = res_final["text"]
        time.sleep(5)
        print("------------------r" + str(count) + " completed--------------")
        count += 1

    # here finally I need to pass to get final RCA
    combine_RCA_PMP = """
        I have possible RCAs (Root Cause Analyses) for a given user query, each outlining different potential reasons. As an RCA expert, your task is to review both RCAs, identify overlaps or complementary insights, and synthesize them into a single, comprehensive, and refined RCA that best explains the root cause of the occurance. Ensure the final RCA is logically structured, eliminates redundancies, and clearly articulates the consolidated reasoning.The query might be a report of an issue, a question about product functionality, or any other inquiry related to managing inventory.
        Here are the two RCAs: 
        RCA 1:{rc1}
        ------------
        RCA 2:{rc2}
        ------------
        Isse:{issue}

        Output format:
        Root Cause Analysis (RCA):
        1)Clearly identify the fundamental underlying cause(s) of the query.
        2)Explain the chain of events or factors that led to the problem.
        3)Support your analysis with evidence from the provided context and logs.
        4)If multiple contributing factors exist, clearly delineate each one and their interaction.

        Workaround(s):
        1)Provide immediate, temporary solutions or steps that can be taken to mitigate the impact of the issue in the short term.
        2)Explain how each workaround helps alleviate the problem.
        3)Consider practicality and ease of implementation.

        Permanent Fix(es):
        1)Outline the long-term solutions required to address the root cause(s) and prevent the issue from recurring.
        2)Provide specific and actionable steps for implementing the permanent fix(es).
        3)Consider the resources, time, and potential impact of implementing these fixes.

        Output Requirements:
        Present your analysis, workarounds, and permanent fixes in a clear, well-structured, and easily understandable manner for a non-expert consumer. Include relevant artifacts where appropriate to illustrate your points.
        """
    combine_temp = PromptTemplate.from_template(combine_RCA_PMP)
    llm_chain_combine = LLMChain(llm=llm, prompt=combine_temp)
    res_combine = llm_chain_combine.invoke(
            {
                "rc1": solution_recommendation["r1"],
                "rc2": solution_recommendation["r2"],
                "issue":question1,
            }
        )
    print("\n------------------Final combined answer--------\n\n")
    print(res_combine["text"])
    CP.remove_tempfiles()
    return res_combine["text"]