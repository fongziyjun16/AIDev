from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

output_parser = DatetimeOutputParser()
template = """
    Answer the users question:

    {question}

    {format_instructions}
"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)
# print(prompt)
# print(prompt.format(question="around when was bitcoin founded?"))
llm = OpenAI()
# chain = prompt | llm
# output = chain.invoke({"question": "around when was bitcoin founded?"})
# print(output)
# output = output_parser.parse(output)
# print(output)
chain = prompt | llm | output_parser
result = chain.invoke({"question": "around when was bitcoin founded?"})
print(result)

