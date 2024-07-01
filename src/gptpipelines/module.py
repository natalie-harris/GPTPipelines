from abc import ABC, abstractmethod
import pandas as pd
import time
from gptpipelines.helper_functions import get_incomplete_entries, truncate, get_unique_columns_and_dtypes
import inspect
import warnings
import logging
import textract
from pdfminer.high_level import extract_text

class Module(ABC):
    """
    An abstract base class for a pipeline module.
    
    This class defines the structure for modules that can be added to a GPTPipeline
    for processing data.

    Attributes
    ----------
    pipeline : GPTPipeline
        Reference to the GPTPipeline instance that the module is part of.
    """

    def __init__(self, pipeline, name=None):
        """
        Initialize a Module instance.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        """

        self.pipeline = pipeline
        self.name = name

    @abstractmethod
    def process(self):
        """
        Abstract method to process input data through the module.

        This method must be implemented by subclasses.
        """

        pass

"""
Valve Module is placed between file df and text df
It limits the amount of texts in text df to n texts, to make sure we don't use up all our memory

Text df automatically deletes texts that are processed (unless specified to save to disk by user)

Internal State:
Max files to read from
Max files that can be in output_df at a time
Number of files read
Number of unprocessed files currently in output_df
"""

class Valve_Module(Module):
    """
    A module to limit the number of texts processed to prevent memory overflow.

    This module is placed between the file DataFrame and text DataFrame and manages
    the flow of texts to ensure that the pipeline does not exceed memory limitations.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    max_files_total : int
        The maximum number of files to read from the input.
    max_files_at_once : int
        The maximum number of files that can be in the output DataFrame at a time.
    current_files : int
        The current number of unprocessed files in the output DataFrame.
    total_ran_files : int
        The total number of files processed so far.
    input_df : pd.DataFrame
        The input DataFrame containing file information.
    output_df : pd.DataFrame
        The output DataFrame where texts are stored.
    """

    def __init__(self, pipeline, files_list_df_name, text_list_df_name, max_at_once=None, num_texts_to_analyze=None, name=None, ocr_language='eng', pdf_method='pdftotext', min_pdf_extract_length=None):
        """
        Initializes a Valve_Module instance.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        num_texts : int
            The maximum number of texts to process in total.
        max_at_once : int, optional
            The maximum number of texts to hold in memory at once. Defaults to 0, which is treated as no limit.
        """

        super().__init__(pipeline, name=name)

        self.input_df_name = files_list_df_name
        self.output_df_name = text_list_df_name
        self.input_df = pipeline.get_df(files_list_df_name)
        self.output_df = pipeline.get_df(text_list_df_name)

        self.ocr_language=ocr_language
        self.pdf_method=pdf_method
        self.min_pdf_extract_length=min_pdf_extract_length
        self.verbose_extraction=self.pipeline.verbose_text_extraction_output

        # Make sure we don't try to access files that don't exist
        self.max_files_total = num_texts_to_analyze
        self.max_files_at_once = max_at_once
        files_left = self.input_df[self.input_df['Completed'] == 0]['File Path'].nunique()
        if files_left == 0:
            print("There are no files left to be processed.")
        elif self.max_files_total is None or files_left < self.max_files_total:
            file_plural = "file" if files_left == 1 else "files"
            print(f"Only {files_left} unprocessed {file_plural} remaining. Only processing {files_left} {file_plural} on this execution.")
            self.max_files_total = files_left
            if self.max_files_at_once is None or files_left < self.max_files_at_once:
                self.max_files_at_once = files_left

        if max_at_once is not None and max_at_once >= 1:
            self.max_files_at_once = max_at_once
        elif self.max_files_total is not None:
            self.max_files_at_once = self.max_files_total
        else:
            self.max_files_at_once = 1
        self.current_files = 0
        self.total_ran_files = 0

    def process(self):
        """
        Processes input data to limit the number of texts in the output DataFrame.

        Overrides the abstract process method in the Module class.

        Returns
        -------
        bool
            True if processing occurred, indicating that there were texts to process; False otherwise.
        """

        working = False

        # only log verbose output if verbose_extraction is True
        original_logging_level = logging.getLogger().getEffectiveLevel()
        if not self.verbose_extraction:
            logging.getLogger().setLevel(logging.ERROR)

        # get number of files in processing in text df by checking for unique instances of Source File where Completed = 0
        self.current_files = self.output_df[self.output_df['Completed'] == 0]['Source File'].nunique()
        while (self.current_files < self.max_files_at_once and self.total_ran_files < self.max_files_total):

            working = True

            # add one file from files list to text list at a time
            has_unprocessed_files = (self.input_df['Completed'] == False).any()
            if not has_unprocessed_files:
                break

            # Find the index of the first entry where 'Completed' is False
            row_index = self.input_df[self.input_df['Completed'] == False].index[0]
            # Set the 'Completed' feature of that entry to True
            self.input_df.at[row_index, 'Completed'] = 1

            # Get the text at the file referenced in File Path
            entry = self.input_df.loc[row_index]
            path = entry['File Path']
            extension = path[path.rfind("."):]
            if extension in ['.txt', '.text']:
                with open(path, 'r', encoding='utf-8') as file:
                    file_contents = file.read()
            
            # extracting text from pdfs
            elif extension == '.pdf' and self.pdf_method != 'tesseract':
                if self.pdf_method == 'pdfminer':
                    file_contents = extract_text(path)
                else:
                    file_contents = textract.process(path, method=self.pdf_method)
                if self.min_pdf_extract_length is not None and len(file_contents) < self.min_pdf_extract_length:
                    file_contents = textract.process(path, method='tesseract', language=self.ocr_language)
            elif extension == '.pdf':
                file_contents = textract.process(path, method=self.pdf_method, language=self.ocr_language)
            
            elif extension in ['gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']:
                file_contents = textract.process(path, language=self.ocr_language)
            else:
                file_contents = textract.process(path)

            new_entry = [path, file_contents, 0]
            self.output_df.loc[len(self.output_df)] = new_entry
            # self.output_df = pd.concat([self.output_df, new_entry])
            self.total_ran_files += 1

            # time.sleep(1)
            self.current_files = self.output_df[self.output_df['Completed'] == 0]['Source File'].nunique()

            # print(f"Output df: [[[\n{self.output_df}\n]]]")

        # print(f"{self.current_files} < {self.max_files_at_once};\t\t{self.total_ran_files} < {self.max_files_total}")

        # set logging back to what it was
        logging.getLogger().setLevel(original_logging_level)

        return working

"""
LLM Modules take in a dataframe as input and write to a dataframe as output. 
Two Types of Input Dataframe Format:
1 - Multiple System Prompts: System Prompt | User Prompt | Examples | Complete
2 - Single System Prompt: User Prompt | Complete (System Prompt and Examples are provided elsewhere in module setup, and are applied the same to every user prompt)

NOTE: allow for custom Complete feature name in case multiple modules are accessing the same df
"""

class LLM_Module(Module):
    """
    An abstract base class for GPT modules.

    This class extends Module to define a structure for modules that interact with
    LLM models for processing text data.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    input_df_name : str
        The name of the input DataFrame.
    output_df_name : str
        The name of the output DataFrame.
    model : str, optional
        The GPT model to use.
    context_window : int, optional
        The context window size for the GPT model.
    safety_multiplier : float, optional
        A multiplier to adjust the maximum token length for safety.
    """

    def __init__(self, pipeline, input_df_name, output_df_name, model=None, context_window=None, safety_multiplier=None):
        """
        Initializes a GPT_Module instance.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        input_df_name : str
            The name of the input DataFrame.
        output_df_name : str
            The name of the output DataFrame.
        model : str, optional
            The GPT model to use. Default is None.
        context_window : int, optional
            The context window size for the GPT model. Default is None.
        safety_multiplier : float, optional
            A multiplier to adjust the maximum token length for safety. Default is None.
        delete : bool, optional
            Whether to delete entries from the input DataFrame after processing. Default is False.
        """

        super().__init__(pipeline)

        #df config
        self.input_df_name = input_df_name
        self.output_df_name = output_df_name

        self.model = model
        self.context_window = context_window
        self.safety_multiplier = safety_multiplier

    @abstractmethod
    def process(self):
        """
        Abstract method to process input data through the GPT model.

        This method must be implemented by subclasses.
        """

        pass

class ChatGPT_Module(LLM_Module):
    """
    A module designed to process texts through a ChatGPT model.

    This module takes input data from a specified DataFrame, processes it through a ChatGPT model,
    and outputs the results to another DataFrame.

    Attributes
    ----------
    Inherits all attributes from the GPT_Module class.

    prompt : str
        The GPT prompt to be used for all entries.
    injection_columns : list of str
        Columns from the input DataFrame whose values are injected into the prompt.
    examples : list
        A list of examples provided to the GPT model for context.
    temperature : float, optional
        The temperature setting for the GPT model. Default is None.
    max_chunks_per_text : int, optional
        The maximum number of chunks into which the input text is split. Default is None.
    timeout : int, optional
        The timeout in seconds for GPT model requests. Default is None.
    loop_function: function, optional
        When supplied, ChatGPT module loops through each chunk in input_text and runs loop_function with each ChatGPT response until loop_function returns True, max_chunks_per_text is reached, or all chunks are evaluated. Output to output_text_column is the last received ChatGPT response, or all received responses wrapped in brackets if wrap==True.
    wrap : bool, optional
        False by default. When true, each response is wrapped in brackets and all responses are stored in one line in the output df. Each response is labelled with the number of the text chunk that prompted it.
    wrap_label : str, optional
        When supplied, prepend wrap_label to the wrapped responses output_response_column string.
    input_text_column : str
        The name of the column in the input DataFrame containing the text to be processed.
    input_completed_column : str
        The name of the column in the input DataFrame that marks whether the entry has been processed.
    output_text_column : str
        The name of the column in the output DataFrame for storing text.
    output_response_column : str
        The name of the column in the output DataFrame for storing the GPT model's response.
    output_completed_column : str
        The name of the column in the output DataFrame that marks whether the entry has been processed.
    """

    def __init__(self, pipeline, input_df_name, output_df_name, prompt, end_message="", injection_columns=[], examples=[], model=None, context_window=None, temperature=None, safety_multiplier=None, max_chunks_per_text=None, timeout=None, loop_function=None, wrap=False, wrap_label=None, include_user_message=True, get_log_probabilities=False, input_text_column='Full Text', input_completed_column='Completed', output_text_column=None, output_response_column='Response', output_log_probabilities_column=None, output_completed_column='Completed'):
        """
        Initializes a ChatGPT_Module instance with specified configuration.

        Parameters
        ----------
        Inherits all parameters from the GPT_Module class and introduces additional parameters for ChatGPT module configuration.
        """
        
        super().__init__(pipeline=pipeline,input_df_name=input_df_name,output_df_name=output_df_name, model=model, context_window=context_window,safety_multiplier=safety_multiplier)

        self.max_chunks_per_text = max_chunks_per_text
        self.temperature=temperature
        self.timeout=timeout
        
        self.input_text_column = input_text_column
        self.input_completed_column = input_completed_column
        self.output_text_column = output_text_column or input_text_column
        self.output_response_column = output_response_column
        self.output_completed_column = output_completed_column
        self.output_log_probabilities_column = output_log_probabilities_column or f"{self.output_response_column} Log Probabilities"

        

        # important gpt request info
        self.prompt = prompt
        self.end_message=end_message
        self.examples = examples
        self.injection_columns = injection_columns
        self.loop_function = loop_function
        self.wrap = wrap
        self.wrap_label = wrap_label
        self.get_log_probabilities = get_log_probabilities
        self.include_user_message = include_user_message

    def setup_dfs(self):
        """
        Sets up the input and output DataFrames based on module configuration.

        Returns
        -------
        bool
            True if setup is successful, False otherwise.
        """

        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_df = self.pipeline.get_df(self.output_df_name)

        if self.input_text_column not in self.input_df.columns and self.include_user_message:
            return False
        elif self.input_completed_column not in self.input_df.columns:
            return False

        features_dtypes = self.pipeline.dfs[self.input_df_name][0].dtypes
        features_with_dtypes = list(features_dtypes.items())

        features = []
        dtypes = []

        # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            if feature != self.input_completed_column and feature != self.input_text_column:
                features.append(feature)
                dtypes.append(dtype)

        for feature, dtype in zip(features, dtypes):
            self.pipeline.dfs[self.output_df_name][0][feature] = pd.Series(dtype=dtype)

        self.pipeline.dfs[self.output_df_name][0][self.output_text_column] = pd.Series(dtype="string")
        self.pipeline.dfs[self.output_df_name][0][self.output_response_column] = pd.Series(dtype="string")
        self.pipeline.dfs[self.output_df_name][0][self.output_completed_column] = pd.Series(dtype="int")
        
        if self.get_log_probabilities:
            self.pipeline.dfs[self.output_df_name][0][self.output_log_probabilities_column] = pd.Series(dtype="object")

        return True

    def process(self):
        """
        Processes the input DataFrame through the ChatGPT model based on the module configuration.

        Overrides the abstract process method in the GPT_Module class.

        Returns
        -------
        bool
            True if processing occurred, indicating that there were texts to process; False otherwise.
        """

        working = False

        input_df = self.pipeline.get_df(self.input_df_name)
        output_df = self.pipeline.get_df(self.output_df_name)
        incomplete_df = get_incomplete_entries(input_df, self.input_completed_column)

        if len(incomplete_df) <= 0:
            return working

        for entry_index in incomplete_df.index:
            entry = input_df.iloc[entry_index]
            text = ""
            if self.include_user_message:
                text = entry[self.input_text_column]

            injections = []
            for column in self.injection_columns:
                injections.append(entry[column])

            # print(truncate(text, 49))

            responses = self.pipeline.process_text(self.prompt, text, self.end_message, injections, self.model, self.context_window, self.temperature, self.examples, self.timeout, self.safety_multiplier, self.max_chunks_per_text, self.loop_function, self.wrap, get_log_probabilities=self.get_log_probabilities)

            if len(responses) > 0 and self.wrap == True:
                r_prompt = responses[0][0]
                r_text = responses[0][1]
                r_examples = responses[0][2]

                response = ""
                log_probs = ""
                if self.wrap_label is not None:
                    response = f"{self.wrap_label}:\n"
                for i in range(len(responses)):
                    next_response = responses[i][3]
                    next_log_probs = responses[i][4]

                    next_response = f"Text #{i+1} response: [{next_response}]\n"
                    next_log_probs = f"{next_log_probs}\n"

                    response += next_response
                    log_probs += next_log_probs



                # print(truncate(response, 49))

                new_responses = [(r_prompt, r_text, r_examples, response, log_probs)]
                responses = new_responses

            # We don't need to include system message or examples for singleprompt module since they are static            
            for system_message, chunk, examples, response, log_probs in responses:
                # Assuming 'entry' is a Series, convert it to a one-row DataFrame
                new_entry_df = entry.to_frame().transpose().copy()
                
                # Drop the unnecessary columns
                if self.include_user_message:
                    new_entry_df = new_entry_df.drop(columns=[self.input_text_column, self.input_completed_column])
                else:
                    new_entry_df = new_entry_df.drop(columns=[self.input_completed_column])

                # Add the new data
                new_entry_df[self.output_text_column] = chunk
                new_entry_df[self.output_response_column] = response
                new_entry_df[self.output_completed_column] = 0
                if self.get_log_probabilities:
                    new_entry_df[self.output_log_probabilities_column] = log_probs
                
                # Identify the next index for output_df
                next_index = len(output_df)
                
                # Iterate over columns in new_entry_df to add them to output_df
                for col in new_entry_df.columns:
                    # Ensure the value type matches the column type in output_df
                    if pd.api.types.is_string_dtype(output_df[col]):
                        output_df.at[next_index, col] = str(new_entry_df[col].values[0])
                    else:
                        output_df.at[next_index, col] = new_entry_df[col].values[0]

            if len(responses) != 0:
                input_df.at[entry_index, self.input_completed_column] = 1
                working = True

        return working     

"""
Code Modules can take in zero or more dataframes as input and write to multiple dataframes as output. They can be in any format
"""
class Code_Module(Module):
    """
    A module for executing custom code as part of the pipeline.

    This module allows for the execution of arbitrary Python functions, facilitating
    custom data processing or transformation within the pipeline.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    code_config : various
        Configuration data or parameters for the custom code execution.
    process_function : function
        The custom function to be executed by the module.
    """

    def __init__(self, pipeline, process_function, input_df_names=[], output_df_names=[]):
        """
        Initializes a Code_Module instance with specified custom code and configuration.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        code_config : various
            Configuration data or parameters for the custom code execution.
        process_function : function
            The custom function to be executed by the module.
        """

        super().__init__(pipeline=pipeline)
        self.process_function = process_function
        self.input_df_names = input_df_names
        self.output_df_names = output_df_names
        self.func_args = {}

    def setup_dfs(self):
        """
        Prepares and validates the input and output DataFrames for the code module.

        This method inspects the parameters of the user-defined process function to determine if `input_dfs` 
        and/or `output_dfs` are expected. It then attempts to prepare these DataFrame dictionaries based on the 
        DataFrame names provided at module initialization. If any specified DataFrames are missing, it issues a warning 
        and returns False to indicate unsuccessful setup.

        Returns
        -------
        bool
            Returns True if all specified DataFrames are successfully prepared and assigned. 
            Returns False if any specified DataFrames are missing, indicating that the setup was unsuccessful.

        Notes
        -----
        - The method utilizes a pipeline-level function `_prepare_dfs` to gather the DataFrames by their names. 
        This function should return a dictionary of DataFrames if all specified names are found, or `None` if any 
        are missing.
        - Warnings are issued if the user's function expects `input_dfs` or `output_dfs` but the respective DataFrame 
        names were not specified at module addition, or if the specified DataFrame names do not exist in the pipeline.

        Examples
        --------
        >>> # Assuming 'input_df_names' were specified as ['sales_data'] during module initialization
        >>> # and 'sales_data' DataFrame exists in the pipeline
        >>> module.setup_df()
        True

        >>> # Assuming 'input_df_names' were specified as ['missing_data'] during module initialization,
        >>> # but 'missing_data' DataFrame does not exist in the pipeline
        >>> module.setup_df()
        UserWarning: Specified input DataFrame(s) 'missing_data' not found in pipeline. Please ensure they are created before running process().
        False
        """

        # Inspect the process_function parameters right in the __init__ method
        params = inspect.signature(self.process_function).parameters

        # Check for 'input_dfs' parameter and prepare if specified
        if 'input_dfs' in params:
            if not self.input_df_names:
                warnings.warn("You've requested 'input_dfs' in your function but did not specify any input_df_names when adding the code module.",
                            UserWarning)
                self.func_args['input_dfs'] = {}
            else:
                input_dfs = self.pipeline._prepare_dfs(self.input_df_names, 'input')
                if input_dfs is None:  # Missing one or more specified input_dfs
                    return False
                self.func_args['input_dfs'] = input_dfs

        # Check for 'output_dfs' parameter and prepare if specified
        if 'output_dfs' in params:
            if not self.output_df_names:
                warnings.warn("You've requested 'output_dfs' in your function but did not specify any output_df_names when adding the code module.",
                            UserWarning)
                self.func_args['output_dfs'] = {}
            else:
                output_dfs = self.pipeline._prepare_dfs(self.output_df_names, 'output')
                if output_dfs is None:  # Missing one or more specified output_dfs
                    return False
                self.func_args['output_dfs'] = output_dfs

        return True

    def process(self):
        """
        Executes the custom code function provided during initialization.

        Overrides the abstract process method in the Module class.

        Returns
        -------
        bool
            The return value of the custom process_function, typically True if processing occurred and False otherwise.
        """

        # NOTE: In user guides, be sure to explain that in order to receive an easy-to-access list of requested dfs, 
        # the user function has to include either/both 'input_dfs' or 'output_dfs' EXACTLY in the the parameters
        # list, and include a proper list of its associated df names in add_code_module.

        # Dynamically call the user's process function with the prepared arguments
        result = self.process_function(self.pipeline, **self.func_args)
        return result
    
class Apply_Module(Module):
    """
    Module designed to iterate through each row in input_df and give requested features from row to an apply_function, which returns features (must be unique from features in input_df) to output_df. Copies all features from input_df into output_df.
    """

    def __init__(self, pipeline, apply_function, input_df_name, output_df_name, input_columns={}, output_columns=[], input_completed_column='Completed', output_completed_column='Completed'):
        super().__init__(pipeline=pipeline)
        self.process_function = apply_function
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.input_df_name = input_df_name
        self.output_df_name = output_df_name
        self.input_completed_column = input_completed_column
        self.output_completed_column = output_completed_column
        self.func_args = {}

    def setup_dfs(self):
        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_df = self.pipeline.get_df(self.output_df_name)

        # Must have at least one output column or else nothing would be added to pipeline
        if len(self.output_columns) <= 0:
            return False

        # Input df must contain all requested input columns
        for input_column in self.input_columns:
            if input_column not in self.input_df.columns or self.input_completed_column not in self.input_df.columns:
                return False
            
        # Requested output features must be unique from input columns so as to maintain as much simplicity as possible (as if any of this stuff is simple =O )
        for output_column in self.output_columns:
            if output_column in self.input_df.columns:
                return False

        features_dtypes = self.pipeline.dfs[self.input_df_name][0].dtypes
        features_with_dtypes = list(features_dtypes.items())

        features = []
        dtypes = []

        # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            features.append(feature)
            dtypes.append(dtype)

        for feature, dtype in zip(features, dtypes):
            self.pipeline.dfs[self.output_df_name][0][feature] = pd.Series(dtype=dtype)

        for output_column in self.output_columns:
            self.pipeline.dfs[self.output_df_name][0][output_column] = pd.Series(dtype=object)

        self.pipeline.dfs[self.output_df_name][0][self.output_completed_column] = pd.Series(dtype="int")

        return True

    def process(self):
        working = False

        input_df = self.pipeline.get_df(self.input_df_name)
        output_df = self.pipeline.get_df(self.output_df_name)
        incomplete_input = input_df[input_df[self.input_completed_column] == 0]

        if len(incomplete_input) > 0:
            working = True

        for index, row in incomplete_input.iterrows():
            # Mark as completed in the input DataFrame in-place
            input_df.at[index, self.input_completed_column] = 1

            # Prepare arguments for processing function
            output_features = {}
            if self.input_columns:
                for column, parameter in self.input_columns.items():
                    self.func_args[parameter] = row[column]

                # Execute processing function and handle its output
                output_features = self.process_function(**self.func_args)
            else:
                output_features = self.process_function()

            # print(output_features)

            # Ensure output is a dictionary with at least one key-value pair
            if not isinstance(output_features, dict):
                raise TypeError("Expected return type dictionary, where keys are feature names and values are the return value for that feature.")
            if len(output_features) <= 0:
                raise ValueError("Apply function must always return at least one key-value pair.")

            # Convert dict_items to a list to access the first item
            output_items = list(output_features.items())
            feature_size = len(output_items[0][1])  # Assuming values are iterable and of equal length

            if len(output_features) > 1:
                for output_feature in output_features.values():
                    if len(output_feature) != feature_size:
                        raise ValueError("All return features must contain the same number of instances.")

            for i in range(feature_size):
                specific_output_features = {key: value[i] for key, value in output_features.items()}

                # print(f"SPECIFIC OUTPUT FEATURES: {list(specific_output_features.items())[:20]}")

                # Convert the original row to a dictionary and update with the new features
                row_dict = row.to_dict()
                row_dict.update(specific_output_features)

                # Append updated row to the output DataFrame
                output_df.loc[len(output_df)] = row_dict

        return working

class Filter_Module(Module):
    """
    Module designed to carry a row from the input df into output df only if user-defined function returns True

    Useful for filtering pipelines.
    """

    def __init__(self, pipeline, check_function, input_df_name, output_df_name, input_columns=[], input_completed_column='Completed'): 
        super().__init__(pipeline=pipeline)
        self.check_function = check_function
        self.input_columns = input_columns
        self.input_df_name = input_df_name
        self.output_df_name = output_df_name
        self.input_completed_column = input_completed_column
        self.func_args = {}

    def setup_dfs(self):

        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_df = self.pipeline.get_df(self.output_df_name)

        # Input df must contain all requested input columns
        for input_column in self.input_columns:
            if input_column not in self.input_df.columns or self.input_completed_column not in self.input_df.columns:
                return False

        features_dtypes = self.pipeline.dfs[self.input_df_name][0].dtypes
        features_with_dtypes = list(features_dtypes.items())

        features = []
        dtypes = []

        # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            features.append(feature)
            dtypes.append(dtype)

        for feature, dtype in zip(features, dtypes):
            self.pipeline.dfs[self.output_df_name][0][feature] = pd.Series(dtype=dtype)


        return True

    def process(self):
        working = False

        input_df = self.pipeline.get_df(self.input_df_name)
        output_df = self.pipeline.get_df(self.output_df_name)
        incomplete_input = input_df[input_df[self.input_completed_column] == 0]

        if len(incomplete_input) > 0:
            working = True

        for index, row in incomplete_input.iterrows():
            # Mark as completed in the input DataFrame in-place
            input_df.at[index, self.input_completed_column] = 1

            # Prepare arguments for processing function
            for column, parameter in self.input_columns.items():
                self.func_args[parameter] = row[column]

            # Execute processing function and handle its output
            output_bool = self.check_function(**self.func_args)
            # print(output_features)

            # Ensure output is a dictionary with at least one key-value pair
            if output_bool is None or not isinstance(output_bool, bool):
                raise TypeError("Expected return type boolean.")

            # Convert the original row to a dictionary and update with the new features
            row_dict = row.to_dict()

            # print(row_dict.keys())

            # If check function returns True, we append row to end of output df
            if output_bool:
                output_df.loc[len(output_df)] = row_dict

        return working

class Duplication_Module(Module):
    """
    A module for duplicating entries from an input DataFrame to multiple output DataFrames.

    This module is designed to facilitate the copying of data across different parts of the pipeline, 
    ensuring that data can be processed in parallel or stored for different purposes without altering the original source.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    input_df_name : str
        The name of the input DataFrame from which entries will be duplicated.
    output_df_names : list of str
        The names of the output DataFrames to which entries will be duplicated.
    input_completed_column : str
        The name of the column in the input DataFrame that marks whether the entry has been processed.
    delete : bool
        Whether to delete entries from the input DataFrame after duplication.
    """

    def __init__(self, pipeline, input_df_name, output_df_names, input_completed_column='Completed', delete=False):
        """
        Initializes a Duplication_Module instance with specified configuration for data duplication.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        input_df_name : str
            The name of the input DataFrame from which entries will be duplicated.
        output_df_names : list of str
            The names of the output DataFrames to which entries will be duplicated.
        input_completed_column : str
            The name of the column in the input DataFrame that marks whether the entry has been processed.
        delete : bool
            Whether to delete entries from the input DataFrame after duplication.
        """

        super().__init__(pipeline=pipeline)
        self.input_df_name = input_df_name
        self.output_df_names = output_df_names
        self.input_completed_column=input_completed_column
        self.delete = delete

    def setup_df(self):
        """
        Prepares the input and output DataFrames for the duplication process.

        Returns
        -------
        bool
            True if setup is successful, False otherwise.
        """

        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_dfs = []
        for output_df_name in self.output_df_names:
            self.output_dfs.append(self.pipeline.get_df(output_df_name))
        
        num_features = self.input_df.shape[1]
        if num_features <= 1: # The input df is empty or just has a completed column, so we can't duplicate
            return False
        elif self.input_completed_column not in self.input_df.columns: # Make sure that the completed column is here 
            return False

        features_dtypes = self.input_df.dtypes
        features_with_dtypes = list(features_dtypes.items())
        # print(features_with_dtypes)

        # print(f"FEATURES: {features_with_dtypes}")
        # print(f"{self.input_text_column}")
        # print(f"{self.input_completed_column}")

        features = []
        dtypes = []

         # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            features.append(feature)
            dtypes.append(dtype)

        for output_df in self.output_dfs:
            for feature, dtype in zip(features, dtypes):
                output_df[feature] = pd.Series(dtype=dtype)

        return True
    
    def process(self):
        """
        Duplicates entries from the input DataFrame to each specified output DataFrame.

        Overrides the abstract process method in the Module class.

        Returns
        -------
        bool
            True if duplication occurred, indicating that there were entries to duplicate; False otherwise.
        """

        working = False

        incomplete_entries = get_incomplete_entries(self.input_df, self.input_completed_column)
        while (len(incomplete_entries) > 0): # while there are any incomplete entries in the input df
            row_index = incomplete_entries.index[0]
            entry = self.input_df.iloc[row_index].values.tolist()
            for output_df in self.output_dfs:
                output_df.loc[len(output_df)] = entry

            if not self.delete:
                self.input_df.at[row_index, self.input_completed_column] = 1
            else:
                self.input_df.drop(row_index, inplace=True)

            working = True

            incomplete_entries = get_incomplete_entries(self.input_df, self.input_completed_column)

        return working

class Combination_Module(Module):
    """
    Module for combining multiple dfs which have the exact same features.
    """

    def __init__(self, pipeline, input_df_names, output_df_name, input_completed_column='Completed', delete=False):
        super().__init__(pipeline=pipeline)
        self.input_df_names = input_df_names
        self.output_df_name = output_df_name
        self.input_completed_column = input_completed_column
        self.delete = delete

    def setup_df(self):
        self.input_dfs = []
        for input_df_name in self.input_df_names:
            self.input_dfs.append(self.pipeline.get_df(input_df_name))
        self.output_df = self.pipeline.get_df(self.output_df_name)

        if len(self.input_dfs) < 1:
            raise ValueError("At least one input df must be passed to Combination Modules")

        features_and_dtypes = get_unique_columns_and_dtypes(self.input_dfs)
        if len(features_and_dtypes) < 2:
            raise ValueError("Input dfs must include at least a Completed column and one other column.")

        for input_df in self.input_dfs:
            input_df_columns = input_df.columns.str.strip()
            input_df_columns_sorted = input_df_columns.sort_values()

            for feature_and_dtype in features_and_dtypes:
                feature = feature_and_dtype[0]
                dtype = feature_and_dtype[1]

                if feature not in input_df_columns_sorted:
                    input_df[feature] = pd.NA

        features = []
        dtypes = []

         # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_and_dtypes:
            features.append(feature)
            dtypes.append(dtype)

        for feature, dtype in zip(features, dtypes):
            # print(feature)
            self.output_df[feature] = pd.Series(dtype=dtype)

        return True

    def process(self):
        working = False

        for input_df in self.input_dfs:
            incomplete_entries = get_incomplete_entries(input_df, self.input_completed_column)
            while (len(incomplete_entries) > 0):
                row_index = incomplete_entries.index[0]
                entry = input_df.iloc[row_index].values.tolist()
                self.output_df.loc[len(self.output_df)] = entry

                if not self.delete:
                    input_df.at[row_index, self.input_completed_column] = 1
                else:
                    input_df.drop(row_index, inplace=True)

                working = True

                incomplete_entries = get_incomplete_entries(input_df, self.input_completed_column)

        return working

class Drop_Module(Module):
    """
    Module for dropping one or more columns from one df to another
    """
    def __init__(self, pipeline, input_df_name, output_df_name, drop_columns=[], input_completed_column='Completed', delete=False):
        super().__init__(pipeline=pipeline)
        self.input_df_name = input_df_name
        self.output_df_name = output_df_name
        self.input_completed_column = input_completed_column
        self.drop_columns = drop_columns
        self.delete = delete

    def setup_dfs(self):
        working = False

        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_df = self.pipeline.get_df(self.output_df_name)

        if self.input_completed_column in self.drop_columns:
            raise ValueError("Completed column must not be included in drop_columns list.")

        columns_to_keep = [col for col in self.input_df.columns if col not in self.drop_columns]

        if len(columns_to_keep) <= 0:
            return working

        for col in columns_to_keep:
            self.output_df[col] = self.input_df[col]

        working = True

        return working

    def process(self):
        working = False

        # Function to get the columns to keep
        def get_columns_to_keep(input_df, drop_columns):
            return [col for col in input_df.columns if col not in drop_columns]

        # Get the list of columns to keep
        columns_to_keep = get_columns_to_keep(self.input_df, self.drop_columns)

        incomplete_entries = get_incomplete_entries(self.input_df, self.input_completed_column)

        while len(incomplete_entries) > 0:
            row_index = incomplete_entries.index[0]
            entry = self.input_df.loc[row_index, columns_to_keep].values.tolist()
            self.output_df.loc[len(self.output_df)] = entry

            if not self.delete:
                self.input_df.at[row_index, self.input_completed_column] = 1
            else:
                self.input_df.drop(row_index, inplace=True)

            working = True

            incomplete_entries = get_incomplete_entries(self.input_df, self.input_completed_column)

        return working