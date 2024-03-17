# NLP-Project-Spam-detector
AntiSpam Filter

The project aims to develop a library capable of analyzing received emails. In particular, the tasks include:

* Training a classifier to identify SPAM emails.
* Identifying the main topics within the SPAM emails in the dataset.
* Calculating the semantic distance between the obtained topics to deduce their heterogeneity.
* Extracting organizations from the non-SPAM emails.

The provided dataset is `spam_dataset.csv`.

You can access the project by referring to the `Email Analysis Project.ipynb` notebook.<br>
Additionally, the Python library `email_analysis_library` contains all the custom functions utilized throughout the project, which are divided into the modules `preprocessing`, `spam_detection`, `lda`, and `ner`.
