# Archaeology Categoriser Assistant.py main app
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import re
import warnings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from util.utility import check_password
from langchain_community.graphs import Neo4jGraph
from langchain.chains.graph_qa.cypher import GraphCypherQAChain


# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(layout="wide")

# Load environment variables
if load_dotenv('.env'):
   #for local development
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
   neo4j_pass = os.getenv('neo4j_pass')
else:
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
   neo4j_pass = os.getenv('neo4j_pass')

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in secrets.")
    st.stop()

# Function to load the FAISS index and known categories
@st.cache_resource(show_spinner=False)
def load_faiss_index_and_categories():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the FAISS index directory
    faiss_index_dir = os.path.join(script_dir, 'faiss_index_book')
    vectorstore = FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)
    # Known categories provided
    known_categories = [
        "Unknown", "Sgraffito", "Carved", "Enamelled", "Glazed", "Moulded",
        "Slip-Cast", "Slip-Trail", "Stamped", "Underglazed", "Transfer-Printed",
        "Overglazed", "Appliqué", "Paddled", "Incised", "Cobalt-Painted"
    ]
    # Deduplicate and normalize categories
    normalized_categories = {cat.strip().lower() for cat in known_categories}
    return vectorstore, normalized_categories


# Function to extract new categories not in the known list
def extract_new_categories(categories_output, known_categories):
    # Normalize the known categories
    known_categories_normalized = {cat.strip().lower() for cat in known_categories}

    # Clean and normalize the output categories
    categories = [
        re.sub(r'[^\w\s]', '', cat.strip()).lower()  # Remove trailing punctuation and convert to lowercase
        for cat in re.split(r',|;', categories_output)
    ]

    # Identify new categories and filter out those with more than 3 words
    new_categories = [
        cat for cat in categories
        if cat and cat not in known_categories_normalized and len(cat.split()) <= 3
    ]

    # Return formatted output
    return ', '.join(new_categories).title() if new_categories else ""


# Function to extract centuries from remark
def extract_centuries(remark):
    pattern = r'(\d{1,2})(?:th|st|nd|rd)?\s*century'
    centuries = re.findall(pattern, remark.lower())
    return [int(c) for c in centuries]

# Function to map centuries to dynasties
def map_centuries_to_dynasties(centuries):
    century_to_dynasty = {
        13: ['Yuan'],
        14: ['Yuan', 'Ming'],
        15: ['Ming'],
        16: ['Ming'],
        17: ['Ming', 'Qing'],
        18: ['Qing'],
        19: ['Qing'],
        20: ['Qing'],  # Up to 1912
    }
    dynasties = set()
    for c in centuries:
        dynasties.update(century_to_dynasty.get(c, []))
    return list(dynasties)

# Cached function to initialize the OpenAI LLM
@st.cache_resource
def get_llm():
    return OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Connect to neo4j aura cloud instance
graph = Neo4jGraph(
    url = "neo4j+s://28dc5915.databases.neo4j.io:7687",
    username = "neo4j",
    password = neo4j_pass
)


# Streamlit app
def classification_app():
    # Do not continue if check_password is not True.
    if not check_password():
        st.stop()

    st.title("Archaeology Categoriser Assistant")

    st.write("Please upload a file with a 'Remarks' column containing text descriptions of the archaeological artefacts.")
    st.write("This assistant will help categorise the artefacts based on their decoration techniques.")

    # Load the FAISS index and known categories
    vectorstore, known_categories = load_faiss_index_and_categories()

    # Initialize the OpenAI LLM
    llm = get_llm()


    # Create a custom prompt template

    prompt_template = """

    You are an archaeology expert specializing in Chinese ceramics decoration techniques.

    Based on the given remark and knowledge base context, classify only the decoration techniques used in the ceramic piece.

    Instructions:

    Consider whether the context is relevant to the remark; it may or may not be.
    Exclude color descriptions, time periods, materials, shapes, forms, or structural elements (e.g., "Blue And White," "19th Century," "Porcelain," "Everted Rim," "Flat Base").
    If there is **any** mention of glaze in the remark (e.g., "Black Glaze," "thin glaze"), classify them as 'Glazed','Underglazed' or 'Overglazed' based on context.
    If the remark is blank or lacks sufficient information, classify as Unknown. 
    Do not classify as Unknown if the remark already mentions a decoration technique in the Decoration Technique Categories list below.
    Use only the exact terms from the Decoration Categories List as classification labels. Only suggest new categories if you strongly believe they apply.
    Provide reasoning by referring to the remark and relevant knowledge base context, if the knowledge base is deemed to be relevant.

    Decoration Technique Categories List:

    1.Sgraffito: The decorative technique of scratching through one surface layer, such as on a wall, slip on a pot, or the surface of glass, so as to reveal a layer beneath. For non-decorative, casual, or defacing marks on walls or other surfaces ancient or modern, use "graffiti."
    2.Carved: The act of shaping, marking, or decorating wood, stone, or another material by cutting or incising, typically using tools such as chisels and other blades. It refers to this process as it is applied to small-scale objects or to objects that are not considered art. "Carving" may also be considered a sculpture technique that is employed in the creation of art.
    3.Enamelled: The process of applying a vitreous coating to metal, ceramic, glass, or other surfaces by fusion using heat in a kiln or furnace, with the result of creating a smooth, hard surface.
    4.Glazed: Overlaying or covering with a smooth and lustrous coating, or polishing and burnishing to create a smooth, shiny surface.
    5.Moulded: Giving form to something by use of a mold; usually refers to pressing a material into the mold, as distinct from pouring liquid material into the mold, for which prefer "casting."
    6.Slip-Cast: The process of forming clayware by pouring slip into plaster molds.
    7.Slip-Trail: The application of slip to a piece of pottery with a syringe-like device so as to create slightly raised linear decoration.
    8.Stamped: Marking the surface of an object by applying pressure with a tool, for example, transferring an ink mark to paper or embossing soft clay; also, applying preprinted labels such as postage stamps that substitute for official stamped marks. In bookbinding, distinguished from "blocking," in which pressure is applied by a machine.
    9.Underglazed: In ceramics, the application of color to the surface of a clay body before application of a transparent glaze. (This includes painted decoration techniques such as by applying pigments or paints onto the ceramic surface, either under or over the glaze.)
    10.Transfer-printed: Process of decorating ceramics by transferring the design from prints on another material, often using heat; invented in England in the 1750s. Designs are first engraved and then printed on paper or another material using a special ceramic ink. The paper is then pressed against the surface of the ceramic, enamel, or glass and the design is thus transferred, often with the application of heat. The process is also used for enamels and glass.
    11.Overglazed: Application of a second glaze to a piece of pottery.
    12.Appliqué: Technique of forming a design by applying cut-out pieces of a material to a ground material; generally associated with needleworking, but also used in ceramics, leatherworking, woodworking, and metalworking.
    13.Paddled: Shaped or decorated by striking the clay surface with a paddle, which may be plain or textured, to create patterns or alter the form before firing.
    14.Incised: The process and technique of producing, forming, or tracing a pattern, text, or other usually linear motif by cutting, carving, or engraving.
    15.Cobalt-painted: Decorated with cobalt oxide, which produces a deep blue color under the glaze, commonly seen in blue-and-white wares.

    Here are 3 examples of how you should do the classification:
    <examples>
    Remark: "The vase features blue and white underglaze patterns "
    Categories: Underglazed
    Reasoning: "As 'blue and white underglaze patterns' are mentioned, the decoration category is Underglazed."

    Remark: "Cizhou black and white painted underglazed basin. Design of vertical parallel strings and grass and two thick horizontal line along the lower section of the basin. The design is on the interior against buff brown body. Exterior features brown glaze. Two sherds glued together."
    Categories: Underglazed
    Reasoning: "The most applicable decoration technique is Underglaze because the decoration involves painted designs (vertical strings, grass, and thick horizontal lines) applied against the buff brown body of the basin before the glaze, which is characteristic of the underglaze painting technique. This technique is commonly seen in Cizhou ware and is well-suited to creating detailed, durable patterns under a transparent glaze."

    Remark: "This porcelain bowl has an everted rim and folded lip with lug on exterior."
    Categories: Unknown
    Reasoning: "'Everted rim' refers to shape and form, which are not decoration techniques."
    </example>

    Remark: {question}

    Possible Supporting Evidence from Knowledge Base: {context}

    Provide the classification labels based **mainly** on the remark,use evidence from knowledge base only if relevant to remark and give reasoning.

    Your response should be in the following format:

    Categories: <comma-separated list of decoration techniques> 
    Reasoning: <your reasoning>
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Graph RAG 
    # Extract all the Ceramics nodes in the knowledge graph
    cypher_query = """
    MATCH (c:Ceramic)
    return c
    """
    nodes = graph.query(cypher_query)

    # Create cutom prompt template for Graph RAG
    prompt_template_graph ="""
    Do step by step:
    Step_1: Look at all the Ceramic nodes in the knowledge graph.
    Ceramic Nodes: {nodes}
    Step_2: Look at the ceramic with the following remark in triple backticks, ignoring any multibyte characters in remark.
    ```
    Remark: {remark}
    ```
    Step_3: Based on the remark from Step_2, select most appropriate Ceramic node from Step_1.
    Step_4: Generate a Cypher to query the techniques used in the selected Ceramic nodes from Step_3.
    Step_5: If there was no context generated from Step_4, output Unknown. Otherwise, go Step_6.
    Step_6: Use information retrieved from Step_4 and remark to classify and output **only the techniques** used in the ceramic.
    Step_7: Output the techniques in terms that are comma separated and **not sentence**.
    """

    # Create GraphCypherQAChain for retrieving from graph
    graph_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )





    uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a .xlsx or .csv file.")
                st.stop()

            if 'Remarks' not in df.columns:
                st.error("The uploaded file must contain a 'Remarks' column.")
                st.stop()

            # Let user select additional columns (except 'Remarks' column)
            available_columns = [col for col in df.columns if col != 'Remarks']
            columns_to_include = st.multiselect("Select additional columns relevant to categorising decoration techniques:", available_columns)

            if st.button("Process File"):
                with st.spinner("Processing the file...🏃💨"):
                    def combine_remarks_with_columns(row, columns):
                        combined_text = row['Remarks']
                        for col in columns:
                            if pd.notna(row[col]):
                                combined_text += f", {col}: {row[col]}"
                            else:
                                combined_text += f", {col}: NaN"
                        return combined_text

                    df = df.astype(str)  # Ensure all are object data type.
                    df['Combined_Remarks'] = df.apply(lambda row: combine_remarks_with_columns(row, columns_to_include), axis=1)

                    # Define classify_decoration_rag within the scope where llm, vectorstore, etc. are accessible
                    def classify_decoration_rag(remark):
                        if not remark.strip():
                            return pd.Series({
                                "Categories": "Unknown",
                                "Reasoning": "No information provided",
                                "NewCategories": "",
                                "Sources": "",
                                "Categories_Graph": "Unknown"

                            })

                        # Extract centuries and map to dynasties
                        centuries = extract_centuries(remark)
                        dynasties = map_centuries_to_dynasties(centuries)

                        # Retrieve documents from the vectorstore
                        retrieved_docs = vectorstore.similarity_search(
                            remark,
                            k=20  # Retrieve more documents to filter later
                        )

                        # Filter documents based on dynasties
                        if dynasties:
                            filtered_docs = [
                                doc for doc in retrieved_docs if doc.metadata.get('Dynasty') in dynasties
                            ]
                            # If no documents remain after filtering, use the top 3 retrieved docs
                            if not filtered_docs:
                                filtered_docs = retrieved_docs[:2]
                        else:
                            filtered_docs = retrieved_docs[:2]

                        # Create the llm_chain with the correct input variables
                        llm_chain = LLMChain(llm=llm, prompt=PROMPT)

                        # Create the StuffDocumentsChain with the llm_chain and specify document_variable_name
                        chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='context')

                        # Run the chain
                        result = chain.run(input_documents=filtered_docs, question=remark)

                        answer = result
                        source_documents = filtered_docs

                        # Parse the answer to extract categories and reasoning
                        categories_match = re.search(r'Categories:\s*(.*)', answer)
                        reasoning_match = re.search(r'Reasoning:\s*(.*)', answer, re.DOTALL)

                        categories = categories_match.group(1).strip() if categories_match else "Unknown"
                        reasoning = reasoning_match.group(1).strip() if reasoning_match else answer


                        
                        # Use graph retriever 
                        # Format the prompt for Graph RAG
                        prompt_graph = PromptTemplate(
                            input_variables = ["remark", "nodes"],
                            template = prompt_template_graph
                        )
                        prompt_graph = prompt_graph.format(remark=remark,nodes=nodes)
                        result_graph = graph_chain.run(prompt_graph)
                        result_graph = str(result_graph)    
                        categories_graph = result_graph                 

                        # Append categories from RAG and Graph RAG together
                        categories_comb = categories + ", " + categories_graph

                        # Extract new categories not in known list
                        new_categories = extract_new_categories(categories_comb, known_categories)

                        # Extract sources from the source documents
                        sources = []
                        for doc in source_documents:
                            page = doc.metadata.get('page', 'Unknown Page')
                            sources.append(f"Page {page}")

                        sources_str = "; ".join(sources)

                        return pd.Series({
                            "Categories": categories,
                            "Reasoning": reasoning,
                            "NewCategories": new_categories,
                            "Sources": sources_str,
                            "Categories_Graph": categories_graph

                        })

                    # Apply the classify_decoration_rag function to each remark
                    df[['Categories', 'Reasoning', 'NewCategories', 'Sources', 'Categories_Graph']] = df['Combined_Remarks'].apply(lambda remark: classify_decoration_rag(remark))


                # Gather all new categories found
                new_categories_list = df['NewCategories'].dropna().str.strip().str.lower()

                # Split and flatten the list
                new_categories_list = new_categories_list[new_categories_list != ""].str.split(',').explode()
                new_categories_list = new_categories_list.str.strip().str.title()
                new_categories_set = sorted(set(new_categories_list))

                if new_categories_set:
                    st.session_state.new_categories = ', '.join(new_categories_set)
                else:
                    st.session_state.new_categories = "No new categories found."

                st.session_state.df = df
                output_filename = uploaded_file.name.replace('.xlsx', '_output.xlsx') if uploaded_file.name.endswith('.xlsx') else uploaded_file.name.replace('.csv', '_output.csv')
                st.session_state.output_filename = output_filename

                # Save the processed file
                if uploaded_file.name.endswith('.xlsx'):
                    df.to_excel(output_filename, index=False)
                else:
                    df.to_csv(output_filename, index=False)

        except Exception as e:
            st.error(f"An error occurred: {type(e).__name__} - {str(e)}")

    # Display the previous outputs if the file is already processed
    if "df" in st.session_state:
        st.subheader("New Categories Found")
        st.write(st.session_state.new_categories)

        st.subheader("Top 5 Rows Preview of the Output")
        st.dataframe(st.session_state.df.head())

        st.subheader("New Columns Preview")
        st.dataframe(st.session_state.df[['Remarks', 'Categories', 'Reasoning', 'Sources']].head())

        st.subheader("Download the Processed File")
        if "output_filename" in st.session_state:
            with open(st.session_state.output_filename, "rb") as file:
                st.download_button(label="Download", data=file, file_name=st.session_state.output_filename)

if __name__ == "__main__":
    classification_app()
