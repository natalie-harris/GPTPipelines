from gptpipelines.module import Module, Valve_Module, ChatGPT_Module, Code_Module, Duplication_Module, LLM_Module, Apply_Module
from gptpipelines.chatgpt_broker import ChatGPTBroker
from gptpipelines.helper_functions import truncate, all_entries_are_true
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
import warnings
from datetime import datetime
import os
import glob
import networkx as nx
from asciinet import graph_to_ascii


class GPTPipeline:
    """
    Manages a pipeline for processing data using the ChatGPT API, incorporating various modules
    for specific tasks such as data handling, GPT interactions, and managing data frames.

    Parameters
    ----------
    api_key : str
        The API key required for authenticating requests with the GPT API.
    organization : str, optional
        Organization ID for billing and usage tracking with the OpenAI platform (default is None).
    verbose_chatgpt_api : bool, optional
        If True, enables verbose logging for ChatGPT API interactions (default is False).
    verbose_pipeline_output : bool, optional
        If True, enables verbose logging for pipeline processing output (default is False).

    Attributes
    ----------
    modules : dict
        Maps module names to their respective module instances {Name: module}.
    dfs : dict
        Maps DataFrame names to tuples {Name: (DataFrame, destination folder)}, managing input and output data.
    gpt_broker : ChatGPTBroker
        Handles interactions with the ChatGPT API, utilizing the provided API key.
    LOG : logging.Logger
        Configured logger for the pipeline, capturing and formatting log messages.
    default_vals : dict
        Default configuration values for the pipeline, including settings for API interactions.
    """

    def __init__(self, api_key=None, path_to_api_key=None, organization=None, verbose_chatgpt_api=False, verbose_pipeline_output=False, verbose_text_extraction_output=False, model=None, context_window=None, temperature=0.0, safety_multiplier=0.95, timeout=15):
        """
        Initializes the GPTPipeline with the provided API key.

        This constructor sets up the basic infrastructure required for the pipeline to function,
        including the management of modules, DataFrames, and interactions with the ChatGPT API.

        Parameters
        ----------
        api_key : str
            The API key required for authenticating requests to the GPT API.
        """

        if api_key is not None:
            self.api_key = api_key
        elif path_to_api_key is not None:
            with open(path_to_api_key, "r") as fd:
                self.api_key = fd.read()
        else:
            print("Either the api key or the path to the api key must be specified at GPTPipeline initialization.")
            exit()

        self.modules = {} # {name: module}
        self.dfs = {} # {name: (df, dest_folder)}
        self.gpt_broker = ChatGPTBroker(api_key=self.api_key, organization=organization, verbose=verbose_chatgpt_api)

        # Set up basic configuration for logging
        self.LOG = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.default_model = model
        self.default_context_window = context_window
        self.default_temperature = temperature
        self.default_safety_multiplier = safety_multiplier
        self.default_timeout = timeout
        self.verbose_text_extraction_output = verbose_text_extraction_output

        if not verbose_pipeline_output:
            warnings.formatwarning = lambda message, category, filename, lineno, line=None: f'\033[91m{category.__name__}:\033[0m {message}\n'

    def get_default_values(self):
        """
        Get the default pipeline configuration values.

        Returns
        -------
        dict
            The default configuration values.
        """

        return self.default_model, self.default_context_window, self.default_temperature, self.default_safety_multiplier, self.default_timeout
    
    def set_default_values(self, gpt_model=None, gpt_context_window=None, temperature=None, safety_multiplier=None, timeout=None):
        """
        Set default configuration values.

        Parameters
        ----------
        default_values : dict
            A dictionary of default values to update.
        """

        self.default_model = gpt_model or None
        self.default_context_window = gpt_context_window or None
        self.default_temperature = temperature or None
        self.default_safety_multiplier = safety_multiplier or None
        self.default_timeout = timeout or None

    def _generate_primary_csv(self, folder_path, dest_csv_path=None, csv_file_name='files.csv', default_features={}, file_extensions=None, include_csv=False):
        """
        Generates a CSV file listing files in a folder to be used as the first df in a GPTPipeline.

        This function searches for files of specified file types in a specified folder, compiles their
        file paths, and generates a CSV file with these paths and additional default features. If a CSV file
        with the specified name already exists at the destination path, the function returns `False` without
        creating a new file. Otherwise, it creates a new CSV file, populating it with the default feature values
        and the file paths of the text files found.

        This function uses textract for retrieving text from files -> https://textract.readthedocs.io/en/stable/index.html
        The valid file_extensions are those supported by textract.

        Parameters
        ----------
        folder_path : str
            The path to the folder from which .txt and .text file paths will be collected.
        dest_csv_path : str, optional
            The destination path where the CSV file will be saved. If `None` (default), uses `folder_path`.
        csv_file_name : str, optional
            The name of the CSV file to be created (default is 'files.csv').
        default_features : dict, optional
            A dictionary specifying the default features and their values to include in the CSV file. Each key-value
            pair corresponds to a column name and its default value (default is an empty dict).

        Returns
        -------
        bool
            `True` if a new CSV file was successfully created; `False` if the CSV file already exists at the
            specified destination path.

        Examples
        --------
        Create a CSV file 'data.csv' in '/path/to/destination' directory listing all text files from 
        '/path/to/folder', with additional columns 'feature1' and 'feature2' having default values 'default1' 
        and 'default2', respectively:

        >>> generate_primary_csv('/path/to/folder', '/path/to/destination', 'data.csv', 
                                default_features={'feature1': 'default1', 'feature2': 'default2'})
        True

        Notes
        -----
        The function checks for the existence of the specified CSV file at the beginning and immediately returns
        `False` if the file already exists, ensuring that existing data is not overwritten.
        """

        if dest_csv_path is None:
            dest_csv_path = folder_path

        # Construct the full path for the csv file
        full_csv_path = os.path.join(dest_csv_path, csv_file_name)
        
        # Check if the CSV file already exists
        if os.path.exists(full_csv_path):
            return False

        valid_file_extensions = {
            '.csv': False,
            '.doc': False,
            '.docx': False,
            '.eml': False,
            '.epub': False,
            '.gif': False,
            '.jpg': False,
            '.jpeg': False,
            '.json': False,
            '.html': False,
            '.htm': False,
            '.mp3': False,
            '.msg': False,
            '.odt': False,
            '.ogg': False,
            '.pdf': False,
            '.png': False,
            '.pptx': False,
            '.ps': False,
            '.rtf': False,
            '.tiff': False,
            '.tif': False,
            '.txt': False,
            '.text': False,
            '.wav': False,
            '.xlsx': False,
            '.xls': False
        }
        if file_extensions is None:
            for extension in valid_file_extensions:
                if extension != '.csv' or include_csv == True:
                    valid_file_extensions[extension] = True
        else:
            for extension in file_extensions:

                extension = extension.lower()
                if not extension.startswith('.'):
                    extension = '.' + extension
                
                if extension == '.jpg' or extension == '.jpeg':
                    valid_file_extensions['.jpg'] = True
                    valid_file_extensions['.jpeg'] = True
                elif extension == '.html' or extension == '.htm':
                    valid_file_extensions['.html'] = True
                    valid_file_extensions['.htm'] = True
                elif extension == '.tiff' or extension == '.tif':
                    valid_file_extensions['.tif'] = True
                    valid_file_extensions['.tiff'] = True
                elif extension == '.text' or extension == '.txt':
                    valid_file_extensions['.text'] = True
                    valid_file_extensions['.txt'] = True
                elif extension in valid_file_extensions:
                    valid_file_extensions[extension] = True
                
                if include_csv:
                    valid_file_extensions['.csv'] = True

        # Initialize the DataFrame with 'File Path' and 'Complete' columns first
        columns = ['File Path', 'Completed'] + list(default_features.keys())
        df = pd.DataFrame(columns=columns)

        # Identify all specified files in the folder
        rows_to_add = []
        extensions = [extension for extension, valid in valid_file_extensions.items() if valid is True]
        for extension in extensions:
            for file_path in glob.glob(f"{folder_path}/*{extension}"):
                # Create a new row with default values, setting 'File Path' and 'Complete'
                new_row = {'File Path': file_path, 'Completed': 0}
                new_row.update(default_features)
                rows_to_add.append(new_row)

        # If there are no files, return False
        if not rows_to_add:
            return False

        # Concatenate the new rows to the DataFrame
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

        # Write DataFrame to CSV
        df.to_csv(full_csv_path, index=False)

        return True    

    def _prepare_text_entries(self, df):
        prepared_df = df.copy(deep=True)

        # Iterate over each column in the DataFrame
        for column in prepared_df.columns:
            # Check if column data type is object; if so, it might contain string objects
            if prepared_df[column].dtype == 'object' or prepared_df[column].dtype == 'string':
                # Apply a function to each entry in the column
         
                prepared_df[column] = prepared_df[column].apply(lambda x: self._process_string_entry(x) if isinstance(x, str) else x)
        
        return prepared_df

    def _process_string_entry(self, entry):
        # Replace newline characters with literal '\n'
        entry = entry.replace('\n', '\\n')
        # Escape double quotes
        entry = entry.replace('"', '\"')
        # Surround the string with double quotes
        entry = f'"{entry}"'
        return entry
        
    def import_texts(self, folder_path, file_name="files.csv", num_texts_to_analyze=None, files_list_df_name="Files List", text_list_df_name="Text List", files_list_dest_folder=None, text_list_dest_folder=None, generate_text_csv=True, text_csv_dest_folder=None, max_files_at_once=None, file_extensions=None, ocr_language='eng', pdf_method='pdfminer', min_pdf_extract_length=None):
        """
        Import texts from a CSV file and populate DataFrames for file and text lists.

        This function uses textract for retrieving text from files -> https://textract.readthedocs.io/en/stable/index.html
        Some file types utilize OCR (tesseract-ocr), and some of these files allow for specifying a language -> https://textract.readthedocs.io/en/stable/python_package.html#additional-options
        Find valid language codes here -> https://github.com/tesseract-ocr/tessdoc/blob/main/Data-Files-in-different-versions.md

        Parameters
        ----------
        path : str
            The file path to the CSV containing the texts.
        num_texts : int
            The number of texts to import.
        """

        if generate_text_csv is True:
            text_csv_dest_folder = text_csv_dest_folder or folder_path # if the user doesn't specify where the text csv should go, just put it in the folder with the texts
            self._generate_primary_csv(folder_path=folder_path, dest_csv_path=text_csv_dest_folder, csv_file_name=file_name, file_extensions=file_extensions)

        path = os.path.join(folder_path, file_name)
        
        files_df = pd.read_csv(path, sep=',')
        text_df = pd.DataFrame(columns=["Source File", "Full Text", "Completed"])

        if files_list_dest_folder is not None:
            self.dfs[files_list_df_name] = (files_df, folder_path)
        else:
            self.dfs[files_list_df_name] = (files_df, None)

        if text_list_dest_folder is not None:
            self.dfs[text_list_df_name] = (text_df, folder_path)
        else:
            self.dfs[text_list_df_name] = (text_df, None)

        # Since users shouldn't need to interact directly with the valve module, 
        # It should be ok to automatically generate a name and not let the user change it.
        valve_module_name = " ".join(["Valve Module for", text_list_df_name])

        self.add_module(valve_module_name, Valve_Module(pipeline=self, num_texts_to_analyze=num_texts_to_analyze, files_list_df_name=files_list_df_name, text_list_df_name=text_list_df_name, max_at_once=max_files_at_once, ocr_language=ocr_language, pdf_method=pdf_method, min_pdf_extract_length=min_pdf_extract_length, name=valve_module_name))

    def import_csv(self, name, dest_folder): # dest_path must point to the folder that the csv file is located in
        """
        Import a CSV file into a DataFrame.

        Parameters
        ----------
        name : str
            The name of the DataFrame.
        dest_path : str
            The destination path where the CSV file is located.
        """

        file_path = os.path.join(dest_folder, name)
        df = pd.read_csv(file_path)
        self.dfs[name] = (df, dest_folder)

    def add_module(self, name, module):
        """
        Add a module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the module.
        module : Module
            The module instance to add.
        """

        if not isinstance(module, Module):
            raise TypeError("Input parameter must be a module")
        self.modules[name] = module

    def add_chatgpt_module(self, name, input_df_name, output_df_name, prompt, end_message=None, injection_columns=[], examples=[], model=None, context_window=None, temperature=None, safety_multiplier=None, max_chunks_per_text=None, timeout=None, input_text_column=None, input_completed_column='Completed', output_text_column=None, output_response_column='Response', output_completed_column='Completed'):
        """
        Add a ChatGPT module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the ChatGPT module.
        input_df_name : str
            The name of the input DataFrame.
        output_df_name : str
            The name of the output DataFrame.
        prompt : str
            The prompt to be used by the ChatGPT module.
        injection_columns : list, optional
            Columns from the input DataFrame to inject into the prompt.
        examples : list, optional
            A list of examples to provide context for the GPT model.
        model : str, optional
            The model to use.
        context_window : int, optional
            The context window size for the GPT model.
        temperature : float, optional
            The temperature setting for the GPT model.
        safety_multiplier : float, optional
            The safety multiplier to adjust the maximum token length.
        max_chunks_per_text : int, optional
            The maximum number of chunks into which the input text is split.
        timeout : int, optional
            The timeout in seconds for GPT model requests.
        input_text_column : str, optional
            The name of the column containing input text in the input DataFrame.
        input_completed_column : str, optional
            The name of the column indicating whether the input is completed.
        output_text_column : str, optional
            The name of the column for text in the output DataFrame.
        output_response_column : str, optional
            The name of the column for the GPT response in the output DataFrame.
        output_completed_column : str, optional
            The name of the column indicating whether the output is completed.
        """

        gpt_module = ChatGPT_Module(pipeline=self, input_df_name=input_df_name, output_df_name=output_df_name, prompt=prompt, end_message=end_message, injection_columns=injection_columns, examples=examples, model=model, context_window=context_window, temperature=temperature, safety_multiplier=safety_multiplier, max_chunks_per_text=max_chunks_per_text, timeout=timeout, input_text_column=input_text_column, input_completed_column=input_completed_column, output_text_column=output_text_column, output_response_column=output_response_column, output_completed_column=output_completed_column)
        self.modules[name] = gpt_module

    def add_code_module(self, name, process_function, input_df_names=[], output_df_names=[]):
        """
        Add a code module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the code module.
        process_function : function
            The function to process data within this module.
        input_df_names : list, optional
            If list has df names (str type) in it, their respective dfs will be passed into `input_dfs` list of DataFrames if they are called in user's process_function().
        output_df_names : list, optional
            If list has df names (str type) in it, their respective dfs will be passed into `output_dfs` list of DataFrames if they are called in user's process_function().
        """

        code_module = Code_Module(pipeline=self, process_function=process_function, input_df_names=input_df_names, output_df_names=output_df_names)
        self.modules[name] = code_module

    def add_apply_module(self, name, apply_function, input_df_name, output_df_name, input_columns=[], output_columns=[], input_completed_column='Completed', output_completed_column='Completed'):
        apply_module = Apply_Module(pipeline=self, apply_function=apply_function, input_df_name=input_df_name, output_df_name=output_df_name, input_columns=input_columns, output_columns=output_columns, input_completed_column=input_completed_column, output_completed_column=output_completed_column)
        self.modules[name] = apply_module

    def add_duplication_module(self, name, input_df_name, output_df_names, input_completed_column='Completed', delete=False):
        """
        Add a duplication module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the duplication module.
        input_df_name : str
            The name of the input DataFrame.
        output_df_names : list
            The names of the output DataFrames.
        input_completed_column : str, optional
            The name of the column indicating whether the input is completed.
        delete : bool, optional
            Whether to delete the input DataFrame after duplication.
        """

        dupe_module = Duplication_Module(pipeline=self, input_df_name=input_df_name, output_df_names=output_df_names, input_completed_column=input_completed_column, delete=delete)
        self.modules[name] = dupe_module

    def add_dfs(self, names, dest_folder=None, features={}):
        """
        Add multiple DataFrames to the pipeline.

        Parameters
        ----------
        names : list of str
            The names of the DataFrames to add.
        dest_folder : str, optional
            The destination path for the DataFrames. A unique suffix will be added based on the DataFrame name. If dest_folder isn't specified, the DataFrame data will not be saved.
        features : dict, optional
            A dictionary specifying the features (columns) and their data types for the new DataFrames.
        """

        for name in names:
            if dest_folder is not None:
                self.add_df(name, dest_folder=dest_folder, features=features)
            else:
                self.add_df(name, features=features)

    def add_df(self, name, dest_folder=None, features={}):
        """
        Add a single DataFrame to the pipeline.

        Parameters
        ----------
        name : str
            The name of the DataFrame to add.
        dest_folder : str, optional
            The destination path for the DataFrame. If dest_folder isn't specified, the DataFrame data will not be saved.
        features : dict, optional
            A dictionary specifying the features (columns) and their data types for the new DataFrame.
        """

        if name in self.dfs:
            print(f"'{name}' label already assigned to another DataFrame. Each DataFrame must have a unique label.")
            exit()

        try:
            df = pd.DataFrame(columns=[*features])
            if len(features) != 0:
                df = df.astype(dtype=features)
            self.dfs[name] = (df, dest_folder)
        except TypeError:
            print("'Features' format: {'feature_name': dtype, ...}")
            exit()

    def _save_dfs(self):
        # Get the current date and time and format the date and time as a string 'YYYY-MM-DD_HH-MM-SS'
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

        for df_name in self.dfs:
            df = self.dfs[df_name][0]
            dest_folder = self.dfs[df_name][1]

            if dest_folder is None:
                continue

            # build file name and path
            suffix = 0
            full_filename = f"{df_name}_{timestamp}.csv"
            full_path = os.path.join(dest_folder, full_filename)

            # add suffix and keep incrementing it while filename already exists at destination folder
            # this is so that we don't lose data by overwriting existing data or not saving new data
            while os.path.exists(full_path):
                suffix += 1

                full_filename = f"{df_name}_{timestamp}_{suffix}.csv"
                full_path = os.path.join(dest_folder, full_filename)

            prepared_df = self._prepare_text_entries(df)

            print(prepared_df)
                
            # Save the DataFrame to CSV
            prepared_df.to_csv(full_path, index=False)

            print(f"DataFrame {df_name} saved to {full_path}.")

        return

    def process(self):
        """
        Process all texts through the pipeline, connecting modules to their respective DataFrames and executing processing tasks.
        """

        # Put max_texts (or all texts if total < max_texts) texts into primary df (add completed feature = 0)
        # Use multiple GPT by bridging with code module, or just use single GPT module

        # connect all modules to their respective dfs
        # to be efficient, this requires a network to determine which modules to setup_df first, for now we will just loop until all output dfs are finished setting up
        finished_setup = {}
        for module in self.modules:
            if not isinstance(self.modules[module], Valve_Module):
                finished_setup[module] = False
            else:
                finished_setup[module] = True

        while not all_entries_are_true(finished_setup):
            made_progress = False
            for module in self.modules:
                if isinstance(self.modules[module], Valve_Module) and finished_setup[module] is not True:
                    finished_setup[module] = True
                    made_progress = True
                elif isinstance(self.modules[module], ChatGPT_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_dfs()
                    finished_setup[module] = result
                    made_progress = result or made_progress
                elif isinstance(self.modules[module], Duplication_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_df()
                    finished_setup[module] = result
                    made_progress = result or made_progress
                elif isinstance(self.modules[module], Code_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_dfs()
                    finished_setup[module] = result
                    made_progress = result or made_progress

                
            if not made_progress:
                print(finished_setup)
                raise RuntimeError("Some dfs cannot be setup")

        # Set all modules to sequentially process until all of them no longer have any uncompleted processing tasks
        working = True
        while working is True:
            working = False
            for module in self.modules:
                working = self.modules[module].process()

        # save each df if dest_path is specified for it
        self._save_dfs()

    def print_modules(self):
        """
        Print the list of modules currently added to the pipeline.
        """

        print(self.modules)
 
    def _get_printable_df(self, df_name):
        """
        Retrieves a DataFrame from the pipeline's data store and returns a version suitable for printing.
        
        This method fetches the DataFrame associated with the provided name and processes it to ensure
        that its content is displayed correctly when printed. It replaces newline characters
        in string entries with spaces to avoid disrupting the layout of the printed DataFrame.
        
        Parameters
        ----------
        df_name : str
            The name of the DataFrame to retrieve and process for printing.
        
        Returns
        -------
        pandas.DataFrame
            A copy of the specified DataFrame with newline characters in string entries replaced by spaces.
        
        Notes
        -----
        This method does not modify the original DataFrame stored in the pipeline's data store.
        It operates on and returns a copy of the DataFrame, ensuring that the original data remains unchanged.
        """

        df = self.dfs[df_name][0]
        return df.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

    def print_dfs(self, names=None):
        """
        Print the specified DataFrames. If no names are provided, print all DataFrames.

        Parameters
        ----------
        names : list of str, optional
            The names of the DataFrames to print. If empty, all DataFrames are printed.
        """
        
        if names is None:
            for df_name in self.dfs:
                formatted_df = self._get_printable_df(df_name)
                print(f"\n{df_name}:\n{formatted_df}")

                # print('')
        else:
            for df_name in names:
                formatted_df = self._get_printable_df(df_name)
                print(f"\n{df_name}:\n{formatted_df}")

    def print_df(self, name, include_path=False):
        """
        Print a single DataFrame and optionally its destination path.

        Parameters
        ----------
        name : str
            The name of the DataFrame to print.
        include_path : bool, optional
            Whether to include the destination path in the output.
        """

        formatted_df = self._get_printable_df(name)

        if include_path is False:
            print(formatted_df)
        else:
            print(formatted_df)
            print(self.dfs[name][1])

    # return a df
    def get_df(self, name, include_path=False):
        """
        Retrieve a single DataFrame and optionally its destination path.

        Parameters
        ----------
        name : str
            The name of the DataFrame.
        include_path : bool, optional
            Whether to include the destination path in the return value.

        Returns
        -------
        pd.DataFrame or tuple
            The requested DataFrame, or a tuple containing the DataFrame and its destination path if include_path is True.
        """

        if include_path is False:
            return self.dfs[name][0]
        else:
            return self.dfs[name]

    def print_files_df(self):
        """
        Print the DataFrame containing the list of files.
        """

        print(self.dfs["Files List"])
    
    def print_text_df(self):
        """
        Print the DataFrame containing the list of texts, truncating the full text to a preview length.
        """

        text_df = self.dfs["Text List"][0]
        for i in range(len(text_df)):
            print(f"Path: {text_df.at[i, 'Source File']}   Full Text: {truncate(text_df.at[i, 'Full Text'], 49)}   Completed: {text_df.at[i, 'Completed']}")

    def _prepare_dfs(self, df_names, df_role):
        """
        Prepares a dictionary of DataFrames based on specified names and their intended role.

        This method attempts to fetch each DataFrame by name from the pipeline's `dfs` attribute. 
        If any specified DataFrame is not found, it records the missing DataFrame names. After 
        attempting to gather all specified DataFrames, if any are missing, it issues a warning and returns `None`.

        Parameters
        ----------
        df_names : list of str
            The names of the DataFrames to be prepared. These names should correspond to keys in the pipeline's `dfs` attribute.
        df_role : str
            A descriptive string indicating the role of the specified DataFrames (e.g., 'input' or 'output'). 
            Used for generating meaningful warning messages.

        Returns
        -------
        dict or None
            If all specified DataFrames are found, returns a dictionary where keys are DataFrame names 
            and values are the DataFrame objects. Returns `None` if any specified DataFrames are missing.

        Raises
        ------
        UserWarning
            Warns the user if any of the specified DataFrame names are not found within the pipeline.

        Examples
        --------
        Assuming the pipeline has a DataFrame registered under the name 'sales_data':

        >>> gpt_pipeline._prepare_dfs(['sales_data'], 'input')
        {'sales_data': <DataFrame object>}

        If a specified DataFrame does not exist:

        >>> gpt_pipeline._prepare_dfs(['nonexistent_data'], 'input')
        UserWarning: Specified input DataFrame(s) nonexistent_data not found in pipeline. Please ensure they are created before running process().
        None
        """

        dfs = {}
        missing_dfs = []
        for df_name in df_names:
            try:
                dfs[df_name] = self.dfs[df_name][0]
            except KeyError:
                missing_dfs.append(df_name)
        
        if missing_dfs:
            missing_str = ", ".join(missing_dfs)
            warnings.warn(f"Specified {df_role} DataFrame(s) {missing_str} not found in pipeline. Please ensure they are created before running process().",
                            UserWarning)
            return None
        return dfs

    def process_text(self, system_message, user_message, end_message="", injections=[], model=None, model_context_window=None, temp=None, examples=[], timeout=None, safety_multiplier=None, max_chunks_per_text=None, module_name=None):
        """
        Process a single text through the GPT broker, handling defaults and injections.

        Parameters
        ----------
        system_message : str
            The system message to send to the GPT model.
        user_message : str
            The user message to process.
        injections : list, optional
            A list of strings to inject into the system message. Useful so that prompts can be somewhat customized for a particular text.
        model : str, optional
            The model to use, None uses the pipeline default.
        model_context_window : int or None, optional
            The context window size, None uses the pipeline default.
        temp : float or None, optional
            The temperature setting for the GPT model, None uses the pipeline default.
        examples : list, optional
            A list of examples to provide context for the GPT model.
        timeout : int or None, optional
            The timeout in seconds for the GPT model request, None uses the pipeline default.
        safety_multiplier : float or None, optional
            The safety multiplier to adjust the maximum token length, None uses the pipeline default.
        max_chunks_per_text : int, optional
            The maximum number of chunks into which the input text is split. Default is all chunks are analyzed.

        Returns
        -------
        list
            A list of tuples containing the processed system message, user message, examples, and GPT response for each chunk.
        """

        # replace defaults
        model = model or self.default_model
        model_context_window = model_context_window or self.default_context_window
        if temp is None or not isinstance(temp, float) or temp > 1.0 or temp < 0.0:
            temp = self.default_temperature
        if timeout is None or not isinstance(timeout, int) or timeout < 0:
            timeout = self.default_timeout
        if safety_multiplier is None or not isinstance(safety_multiplier, float) or safety_multiplier < 0.0:
            safety_multiplier = self.default_safety_multiplier

        # FIXME: process_text needs to check to make sure every variable exists and is of the correct type.
        # If they aren't, then it needs to close the program gracefully so that we don't lose analyzed data.
        if model is None or model_context_window is None:
            print("No model was specified. Please specify a model in either the GPT module or as a GPTPipeline default value.")
            exit()

        # inject our injections as a replacement for multiprompt module
        # allows for doing {{}} for edge case when user wants {} in their prompt without injecting into it
        nonplaceholders_count = system_message.count('{{}}')
        placeholders_count = system_message.count('{}')
        placeholders_count = placeholders_count - nonplaceholders_count

        if len(injections) > 0 and len(injections) == placeholders_count:
            system_message = system_message.format(*injections)
        elif len(injections) != placeholders_count:
            print("Inequivalent number of placeholders in system message and injections. Not injecting into system prompt to prevent errors. If you mean to have curly brace sets in your system prompt ({}), then escape them by wrapping them in another set of curly braces ({{}}).")

        # make sure breaking up into chunks is even possible given system message and examples token length
        static_token_length = self.gpt_broker.get_tokenized_length(system_message, "", model, examples, end_message=end_message)
        if static_token_length >= int(model_context_window * safety_multiplier):
            print(f"The system message and examples are too long for the maximum token length ({int(model_context_window * safety_multiplier)})")
            return ['GPT API call failed.']

        text_chunks = self.gpt_broker.split_message_to_lengths(system_message, user_message, model, model_context_window, examples, end_message, safety_multiplier)
        if max_chunks_per_text is not None:
            text_chunks = text_chunks[0:max_chunks_per_text]

        # setup progress bar
        if module_name is not None:
            description = module_name
        else:
            description = "Processing"
            
        pbar = tqdm(total=len(text_chunks), leave=False, desc=description)

        responses = []
        for chunk in text_chunks:
            response = self.gpt_broker.get_chatgpt_response(self.LOG, system_message, chunk, model, model_context_window, end_message, temp, examples, timeout)
            responses.append((system_message, chunk, examples, response))
            pbar.update(1)
            pbar.refresh()

        return responses
    
    def visualize_pipeline(self):
        # Create a networkx graph
        G = nx.Graph()

        for name, module in self.modules.items():
            class_type = module.__class__
            class_type_str = class_type.__name__ + '\n'
            if issubclass(class_type, Valve_Module):
                G.add_edge('DataFrame\n'+module.input_df_name, class_type_str+name)
                G.add_edge('Valve_Module\n'+name, 'DataFrame\n'+module.output_df_name)
            elif issubclass(class_type, LLM_Module):
                G.add_edge('DataFrame\n'+module.input_df_name, class_type_str+name)
                G.add_edge(class_type_str+name, 'DataFrame\n'+module.output_df_name)
            elif issubclass(class_type, Duplication_Module):
                G.add_edge('DataFrame\n'+module.input_df_name, class_type_str+name)
                for output_df_name in module.output_df_names:
                    G.add_edge(class_type_str+name, 'DataFrame\n'+output_df_name)

        for name, df_zip in self.dfs.items():
            df = df_zip[0]
            dest_folder = df_zip[1]
            if dest_folder is not None:
                G.add_edge('DataFrame\n'+name, 'Storage\n'+dest_folder)
        
        # Print the ASCII representation of the graph
        print(graph_to_ascii(G))
