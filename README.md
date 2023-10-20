# Data Visualisation Tool
1. Preparation: Please make sure that necessary libraries such as matplotlib, scipy, numpy, os, and csv are correctly installed in your Python environment.
2. File and Data Configuration: Specify the CSV files from which data will be extracted by modifying the variables FILE_NAME1 and FILE_NAME2. You may define two data columns for each CSV file to be processed and plotted. Customize the names of the data columns using variables such as DATA_1A, DATA_1B, DATA_2A, and DATA_2B.
3. Plotting Configuration: Customize the appearance of your plots, such as titles and axis labels, by altering the TITLE, Y_AXIS_LABEL, and other relevant variables.
4. Filter Configuration: The code comes with a built-in Butterworth filter. You can apply this filter to your data by setting the corresponding data filter variables to True, e.g., DATA_1A_FILTER = True. Configure the filter's cutoff frequency and order by modifying the CUTOFF and ORDER variables.
5. Running the Application: Execute the Python script. The main function will manage the data extraction, processing, and plotting processes. The extracted data will be processed, filtered, and visualized in a plot displayed at the end of the script execution.
6. Code Customization: Please customize the code further according to your research requirements. Essential functions like extact_data, plot_experiment, and apply_butterworth are available for more advanced manipulations and adaptations.
7. Execution: Run the script in a Python environment, and it should automatically handle data extraction, processing, and visualization based on your configurations.
