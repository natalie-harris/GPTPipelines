# gpt_parser/chatgpt_broker.py
import openai
from openai import OpenAI
import tiktoken
import time
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import json

class ChatGPTBroker:
    """
    A broker class for interacting with the OpenAI API to utilize ChatGPT models.

    This class abstracts away the details of making requests to the OpenAI API,
    handling tokenization of messages, splitting messages to fit model's context window,
    and fetching responses from ChatGPT models.

    Parameters
    ----------
    api_key : str
        The API key used for authenticating requests to the OpenAI API.
    organization : str, optional
        The organization ID for billing and usage tracking on OpenAI's platform.
        Defaults to an empty string, which uses the default organization tied to the API key.
    verbose: bool, optional
        If true, print out verbose output of https requests, warnings/errors, etc. related to the ChatGPT api .
        Defaults to False.

    Attributes
    ----------
    api_key : str
        Stores the API key for the OpenAI API.
    client : openai.OpenAI
        The OpenAI client instance configured with the provided API key and organization.
    verbose: bool
        Stores user's preference for verbose output.
    """

    def __init__(self, api_key, organization=None, verbose=False):
        """
        Initializes the ChatGPTBroker with the given API key and optional organization and verbosity preference.
        """

        self.api_key = api_key
        if organization is not None:
            self.client = OpenAI(api_key=api_key, organization=organization)
        else:
            self.client = OpenAI(api_key=api_key)
        self.verbose=verbose

        if not self.verbose:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.basicConfig(level=logging.INFO, format='%(message)s')

    def get_tokenized_length(self, system_message, user_message, model, examples=[], end_message="", tokenizer=None):
        """
        Calculates the total number of tokens for a given set of messages and examples,
        based on the tokenization process of a specified model.

        Parameters
        ----------
        system_message : str
            The system message or prompt to prepend to the user message.
        user_message : str
            The user message to be tokenized.
        model : str
            The model identifier to use for tokenization, determining how text is split into tokens.
        examples : list of dict, optional
            Additional examples to include in the tokenization, where each example is a dictionary
            containing at least a "content" key. Defaults to an empty list.

        Returns
        -------
        int
            The total number of tokens after tokenizing the combined messages and examples.
        """
        
        total_text = system_message + user_message + end_message

        # Loop through the list of example dictionaries (if provided)
        # and append the content of each example to the input text.
        for example in examples:
            total_text += example["content"]
        
        # Get the encoding (tokenizer) associated with the specified model.
        if tokenizer is not None:
            encoding = tiktoken.encoding_for_model(tokenizer)
        else:
            encoding = tiktoken.encoding_for_model(model)
        
        # Use the encoding (tokenizer) to tokenize the text
        # and then calculate the number of tokens in the tokenized text.
        num_tokens = len(encoding.encode(total_text))
        
        return num_tokens
    
    # safety multipliers limits max message length just in case tiktoken incorrectly splits tokens
    def split_message_to_lengths(self, system_message, user_message, model, max_context_window, examples=[], end_message="", safety_multiplier=1.0, tokenizer=None):
        """
        Splits a message into chunks that fit within a model's maximum context window, considering
        safety multipliers and additional examples.

        Parameters
        ----------
        system_message : str
            The system message to prepend to each chunk.
        user_message : str
            The full user message to be split.
        model : str
            The identifier of the model to be used.
        max_context_window : int
            The maximum number of tokens that can be included in a single request.
        examples : list, optional
            A list of examples to consider for tokenization alongside the messages.
        safety_multiplier : float, optional
            A factor to apply to the max_context_window to reduce the risk of exceeding the token limit.

        Returns
        -------
        list of str
            A list of message chunks, each fitting within the specified token limit.
        """

        if tokenizer is None:
            tokenizer = model

        if safety_multiplier > 1.0:
            safety_multiplier = 1.0
        elif safety_multiplier <= 0:
            safety_multiplier = 0.01

        static_token_length = self.get_tokenized_length(system_message, "", model, examples, end_message=end_message, tokenizer=tokenizer)
        if static_token_length >= max_context_window * safety_multiplier:
            return []

        total_token_length = self.get_tokenized_length(system_message, user_message, model, examples, end_message=end_message, tokenizer=tokenizer)
        if total_token_length <= max_context_window * safety_multiplier:
            return [user_message]
        
        base_multiplier = 4
        max_chunk_tokens = int((max_context_window - static_token_length) * safety_multiplier)
        chunks = []  # Will hold the resulting chunks of text

        # # need to finish and debug this logic later
        i = 0  # Start index for slicing the text
        while i < len(user_message):
            # Calculate the length of a user message chunk
            multiplier = base_multiplier
            new_index = int(max_chunk_tokens * multiplier)

            user_chunk = user_message[i:i+new_index]
            user_chunk_tokens = self.get_tokenized_length('', user_chunk, model, [], tokenizer=tokenizer)
            
            # If the token length exceeds the max allowed, reduce the message length and recheck
            while user_chunk_tokens > max_chunk_tokens:
                multiplier *= 0.95
                new_index = int(max_chunk_tokens * multiplier)
                user_chunk = user_message[i:i+new_index]
                user_chunk_tokens = self.get_tokenized_length('', user_chunk, 'gpt-3.5-turbo', [], tokenizer=tokenizer)
            
            # Save the chunk and move to the next segment of text
            chunks.append(user_chunk)
            i += len(user_chunk)
        
        # else we need to split up the message into chunks. I may have a function that does this in original SBW parser
        return chunks
    
    def get_chatgpt_response(self, LOG, system_message, user_message, model, model_context_window, end_message="", temp=0, get_log_probabilities=False, examples=[], timeout=15, tokenizer=None):
        """
        Fetches a response from ChatGPT based on a user's message and a system message.

        Parameters
        ----------
        system_message : str
            A message that defines the context or instructions for the ChatGPT model.
        user_message : str
            The user's message to which the ChatGPT model will respond.
        model : str
            The identifier of the ChatGPT model to use for generating the response.
        model_context_window : int
            The maximum length (in tokens) that the model can handle in a single prompt.
        temp : float, optional
            The temperature parameter to control the randomness of the response. Defaults to 0.
        examples : list of dict, optional
            A list of additional examples to provide context to the model. Defaults to an empty list.
        timeout : int, optional
            The timeout in seconds for waiting for a response from the API. Defaults to 15.

        Returns
        -------
        str or None
            The generated response from the ChatGPT model, or None if an error occurred.
        """

        if tokenizer is None:
            tokenizer = model

        tokenized_length = self.get_tokenized_length(system_message, user_message, model, examples, tokenizer=tokenizer)
        if tokenized_length > model_context_window:
            logging.info('Prompt too long...')
            return ['Prompt too long...']
        
        # Prepare the messages to send to the Chat API
        new_messages = [{"role": "system", "content": system_message}]
        if len(examples) > 0:
            new_messages.extend(examples)
        new_messages.append({"role": "user", "content": user_message})

        if len(end_message) > 0:
            new_messages.append({"role": "system", "content": end_message})
        
        # Flag to indicate whether a response has been successfully generated
        got_response = False
        
        # Continue trying until a response is generated
        retries = 0
        max_retries = 10
        with logging_redirect_tqdm():
            while not got_response and retries < max_retries:
                try:
                    # Attempt to get a response from the GPT model
                    response = self.client.chat.completions.create(model=model,
                    messages=new_messages,
                    temperature=temp,
                    logprobs=get_log_probabilities,
                    timeout=timeout)
                    
                    # Extract the generated text from the API response
                    choices = response.choices[0]
                    generated_text = choices.message.content
                    log_probs = None
                    if get_log_probabilities:
                        log_probs = choices.logprobs.content
                    got_response = True

                    if get_log_probabilities:
                        formatted_log_probs = {
                            'tokens': [],
                            'log_probs': []
                        }
                        for log_prob in log_probs:
                            formatted_log_probs['tokens'].append(log_prob.token)
                            formatted_log_probs['log_probs'].append(log_prob.logprob)
                        log_probs = json.dumps(formatted_log_probs)

                    return generated_text, log_probs
                    
                except openai.RateLimitError as err:
                    # Handle rate limit errors
                    if 'You exceeded your current quota' in str(err):
                        LOG.info("You've exceeded your current billing quota. Go check on that!")
                        return 'BILLING QUOTA ERROR'
                    num_seconds = 3
                    LOG.info(f"Waiting {num_seconds} seconds due to high volume of {model} users.")
                    time.sleep(num_seconds)
                                    
                except openai.APITimeoutError as err:
                    # Handle request timeouts
                    num_seconds = 3
                    LOG.info(f"Request timed out. Waiting {num_seconds} seconds and retrying...")
                    retries += 1
                    time.sleep(num_seconds)
                    
                except openai.InternalServerError as err:
                    # Handle service unavailability errors
                    num_seconds = 3
                    LOG.info(f"There's a problem at OpenAI's servers. Waiting {num_seconds} seconds and retrying request.")
                    time.sleep(num_seconds)

                except openai.APIError as err:
                    # Handle generic API errors
                    LOG.info("An error occurred. Retrying request.")

        return None