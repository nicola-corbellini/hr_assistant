import json

from typing import List
from pydantic import BaseModel, Field, validator
from langchain.chains import LLMChain
from langchain_core.prompts.prompt import PromptTemplate

from cat.experimental.form import form, CatForm, CatFormState


# Data structure to fill up
class JobDescription(BaseModel):
    first_name: str = Field(description="The applicant's name")
    last_name: str = Field(description="The applicant's last name")
    education: str = Field(description="The applicant's education degree")
    job_position: str = Field(description="The position the human is applying for")
    years_in_the_field: int = Field(description="The number of years the applicant worked as a software developer")
    known_languages: List[str] = Field(description="The list of known languages")
    motivation_letter: str = Field(description="A max. 500 characters short presentation of the user and his or her motivation",
                                   min_length=1, max_length=500)

    # Validator to ensure the job_position is among the available ones
    @validator("job_position")
    def validate_job_position(cls, v):
        valid_positions = ["senior software developer", "junior software developer", "product manager"]
        if v.lower() not in valid_positions:
            raise ValueError(f"{v} is an invalid job position, valid ones are: {', '.join(valid_positions)}")
        return v


# Forms let you control goal oriented conversations
@form
class JobApplicationForm(CatForm):
    description = "useful to gather information about the application for a job position"
    model_class = JobDescription
    start_examples = [
        "I want to apply for a job"
    ]
    stop_examples = [
        "Stop application",
        "I don't want to apply anymore"
    ]
    ask_confirm = True

    def message_incomplete(self):
        # Description of the application using the Pydantic model
        out = f"""Job Application:

        ```json
        {json.dumps(self._model, indent=4)}
        ```
        """
        # Format the missing and invalid fields with a dotted list
        separator = "\n - "
        missing_fields = ""
        if self._missing_fields:  # Property of the CatForm class that stores the fields yet to be filled
            missing_fields = "\nMissing fields:"
            missing_fields += separator + separator.join(self._missing_fields)
        invalid_fields = ""
        if self._errors:  # Property of the CatForm class that store the fields that raised a validation error
            invalid_fields = "\nInvalid fields:"
            invalid_fields += separator + separator.join(self._errors)

        # Method to get a string version of the conversation history
        chat_history = self.cat.stringify_chat_history()

        # Prompt for the LLM
        prompt = f"""Your task is to gather information from the human that is applying for a job position,
        use the information below to assist the human providing useful hints and asking for the missing fields.
        Act like you're doing a job interview.

        {{out}}

        {missing_fields}

        {invalid_fields}

        {chat_history}
        AI: """

        # Use a Langchain LLMChain to ask a completion with the prompt above
        template = PromptTemplate.from_template(prompt)
        extraction_chain = LLMChain(
            prompt=template,
            llm=self.cat._llm,
            verbose=True,
            output_key="output"
        )

        llm_answer = \
        extraction_chain.invoke(input={"out": out})["output"]

        # Return the generated sentence to prompt the user to provide the information to fill the form
        return {
            "output": llm_answer
        }

    def message_wait_confirm(self):
        # Description of the application using the Pydantic model
        out = f"""Summary:

```json
{json.dumps(self._model, indent=4)}
```
Confirm? Yes or no.                
"""
        return {
            "output": out
        }

    def submit(self, form_data):
        # Let's embed the form to perform a search in the memory
        form_embedding = self.cat.embedder.embed_query(json.dumps(form_data))

        # Query the declarative memory and format the results
        declarative_memory = self.cat.memory.vectors.declarative
        memory_documents = declarative_memory.recall_memories_from_embedding(form_embedding)
        memories = [m[0].page_content for m in memory_documents]
        formatted_memories = "\n".join(memories)

        # Prompt
        prompt = f"""You are reviewing a job application. Given the following candidate's profile:
        {form_data}

        please, compare the aforementioned profile with the following resume excerpts
        {formatted_memories}

        then, provide a well explained feedback to the applicant, considering the following:
        - "motivation": a vote on a scale from 0 to 5,
        - "expertise": a vote on a scale from 0 to 5,
        - "comparison": the similarity of the candidate's profile with the resume excerpts on a scale from 0 to 5, 
        - "overall score": A final score considering the previous two. Keep in mind that proficiency in Python is a strong requirement.
        If the user doesn't know Python, the overall score should be 0.

        Communicate the feedback and the final decision to the user: """

        feedback = self.cat.llm(prompt)

        # Return to convo
        return {
            "output": feedback
        }


