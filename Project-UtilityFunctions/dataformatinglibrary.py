import numpy as np

#Libraries for printing tables in readable format
from tabulate import tabulate

#Library for creating an excel sheet
import xlsxwriter

def createExcelFromArray(array, fileName):
    workbook = xlsxwriter.Workbook(fileName)
    worksheet = workbook.add_worksheet()

    row = 0
    for col, data in enumerate(array):
        worksheet.write_row(col, row, data)

    workbook.close()

def printList (list,heading):
    for i in range(0, len(list)): 
        list[i] = str(list[i]) 
    if len(list)>0:
        print(tabulate([i.strip("[]").split(", ") for i in list], headers=[heading], tablefmt='orgtbl')+"\n")
