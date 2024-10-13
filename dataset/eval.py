import os
import csv
import pandas as pd

'''
Entities:
1. employerName
2. employerAddressStreet_name
3. employerAddressCity
4. employerAddressState
5. employerAddressZip
6. einEmployerIdentificationNumber
7. employeeName
8. ssnOfEmployee
9. box1WagesTipsAndOtherCompensations
10. box2FederalIncomeTaxWithheld
11. box3SocialSecurityWages
12. box4SocialSecurityTaxWithheld
13. box16StateWagesTips
14. box17StateIncomeTax
15. taxYear
'''



'''
Description: The fuction yields the standard precision, recall and f1 score metrics

arguments:
    TP -> int
    FP -> int
    FN -> int

returns: float, float, float
'''
def performance(TP, FP, FN):
    
    if (TP+FP) == 0:
        precision = "NaN"
    else:
        precision = TP/float((TP+FP))
        
    if (TP+FN) == 0:
        recall = "NaN"
    else:
        recall = TP/float((TP+FN))
    
    if (recall!="NaN") and (precision!="NaN"):
        f1_score = (2.0*precision*recall)/(precision+recall)
    else:
        f1_score = "NaN"
    
    return precision, recall, f1_score
    
    
    
    
'''
Description: The fuction yields a dataframe containing entity-wise performance metrics

arguments:
    true_labels -> list
    pred_labels -> list
    
returns: pandas dataframe
'''
def get_dataset_metrics(true_labels, pred_labels):
    
    metrics_dict = dict()
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label not in metrics_dict:
            metrics_dict[true_label] = {"TP":0, "FP":0, "FN":0, "Support":0}
        print("true lable: ", true_label)
        print("pred lable: ", pred_label)
        if true_label != "OTHER":
            metrics_dict[true_label]["Support"] += 1
            
            if true_label == pred_label:
                metrics_dict[true_label]["TP"] += 1
            
            elif pred_label == "OTHER":
                metrics_dict[true_label]["FN"] += 1
            
        else:
            if pred_label != "OTHER":
                metrics_dict[pred_label]["FP"] += 1
                print("--------------------------here----------------------------")
           
    df = pd.DataFrame()
    
    for field in metrics_dict:
        precision, recall, f1_score = performance(metrics_dict[field]["TP"], metrics_dict[field]["FP"], metrics_dict[field]["FN"])
        support = metrics_dict[field]["Support"]
        
        if field != "OTHER":
            temp_df = pd.DataFrame([[precision, recall, f1_score, support]], columns=["Precision", "Recall", "F1-Score", "Support"], index=[field])
            print(temp_df)
            df = df.append(temp_df)
    
    return df




'''
Description: The fuction yields a dataframe containing entity-wise performance metrics for a single document
(make sure the doc id is the same)

arguments:
    doc_true -> tsv file with with labels in the last column (8 th column (1-indexed))
    doc_pred -> tsv file with labels in the last column (8 th column (1-indexed)), as predicted by the model
    
returns: list, list
'''
def get_doc_labels(doc_true, doc_pred):

    true_labels = [row[-1] for row in csv.reader(open(doc_true, "r"))]
    pred_labels = [row[-1] for row in csv.reader(open(doc_pred, "r"))]

    return true_labels, pred_labels



'''
Description: The fuction yields a dataframe containing entity-wise performance metrics for all documents
(make sure the doc ids are the same in both the paths)

arguments:
    doc_true -> string (directory containing the ground truth tsv files)
    doc_pred -> string (directory containing the predicted tsv files)
    save -> bool (saves the metrics file in your working directory)
returns: pandas dataframe
'''
def get_dataset_labels(true_path, pred_path, save=False):
    
    y_true, y_pred = [], []
    
    for true_file in os.listdir(true_path):
        print(true_file)
        for pred_file in os.listdir(pred_path):
            print(pred_file)
            if (".tsv" in true_file) and (".tsv" in pred_file):
                if true_file == pred_file:
                    
                    true_file, pred_file = f"{true_path}/{true_file}", f"{pred_path}/{pred_file}"
                    true_labels, pred_labels = get_doc_labels(true_file, pred_file)
                    
                    y_true.extend(true_labels)
                    y_pred.extend(pred_labels)
            
    df = get_dataset_metrics(y_true, y_pred)
    print(df)
    if save == True:
        df.to_csv("eval_metrics.tsv")



if __name__ == "__main__":
    
    # template to run your own evaluation

    doc_true = f"{os.getcwd()}\\dataset\\val_w_ann\\boxes_transcripts_labels"
    doc_pred = f"{os.getcwd()}\\predicted_outputs"

    get_dataset_labels(doc_true, doc_pred, save=False)

        
        
        
    
    
    
