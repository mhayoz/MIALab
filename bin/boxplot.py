import matplotlib.pyplot as plt
import numpy as np
import csv


WhiteMatter_DICE = []
GreyMatter_DICE = []
Hippocampus_DICE = []
Amygala_DICE = []
Thalamus_DICE = []

WhiteMatter_SPCFTY = []
GreyMatter_SPCFTY = []
Hippocampus_SPCFTY = []
Amygala_SPCFTY = []
Thalamus_SPCFTY = []

WhiteMatter_SNSVTY = []
GreyMatter_SNSVTY = []
Hippocampus_SNSVTY = []
Amygala_SNSVTY = []
Thalamus_SNSVTY = []


with open('mia-result/results.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # s = ", "
            # print(f'Column headers are {s.join(row)}')
            line_count += 1
        else:
            #print(f'\tImage ID: {row[0]} | Classlabel: {row[1]} | Dice Coefficient: {row[2]} | Specificity: {row[3]} | Sensitivity: {row[4]}')

            if row[1] == "WhiteMatter":
                WhiteMatter_DICE.append(float(row[2]))
                WhiteMatter_SPCFTY.append(float(row[3]))
                WhiteMatter_SNSVTY.append(float(row[4]))
            elif row[1] == "GreyMatter":
                GreyMatter_DICE.append(float(row[2]))
                GreyMatter_SPCFTY.append(float(row[3]))
                GreyMatter_SNSVTY.append(float(row[4]))
            elif row[1] == "Hippocampus":
                Hippocampus_DICE.append(float(row[2]))
                Hippocampus_SPCFTY.append(float(row[3]))
                Hippocampus_SNSVTY.append(float(row[4]))
            elif row[1] == "Amygdala":
                Amygala_DICE.append(float(row[2]))
                Amygala_SPCFTY.append(float(row[3]))
                Amygala_SNSVTY.append(float(row[4]))
            elif row[1] == "Thalamus":
                Thalamus_DICE.append(float(row[2]))
                Thalamus_SPCFTY.append(float(row[3]))
                Thalamus_SNSVTY.append(float(row[4]))
            line_count += 1




numOfClasses = 5
classnames = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala',
               'Thalamus']

data_DICE = [WhiteMatter_DICE, GreyMatter_DICE, Hippocampus_DICE, Amygala_DICE, Thalamus_DICE]
data_SPCFTY = [WhiteMatter_SPCFTY, GreyMatter_SPCFTY, Hippocampus_SPCFTY, Amygala_SPCFTY, Thalamus_SPCFTY]
data_SNSVTY = [WhiteMatter_SNSVTY, GreyMatter_SNSVTY, Hippocampus_SNSVTY, Amygala_SNSVTY, Thalamus_SNSVTY]

fig, ax = plt.subplots(3,1)
fig.canvas.set_window_title('SVM with rbf kernel evaluation')
fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1,  hspace = 0.6 )



#Plot DICE
bp = ax[0].boxplot(data_DICE, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
# Hide these grid behind plot objects
ax[0].set_axisbelow(True)
ax[0].set_title('Results for Dice Coefficient')
#ax[0].set_xlabel('Label')
ax[0].set_ylabel('Dice Coefficient')

# Set the axes ranges and axes labels
ax[0].set_xlim(0.5, numOfClasses + 0.5)
ax[0].set_ylim(0, 1)
ax[0].set_xticklabels(classnames, rotation=0, fontsize=8)



#Plot SPCFTY
bp = ax[1].boxplot(data_SPCFTY, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax[1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax[1].set_axisbelow(True)
ax[1].set_title('Results for Specificity')
#ax[1].set_xlabel('Label')
ax[1].set_ylabel('Specificity')

# Set the axes ranges and axes labels
ax[1].set_xlim(0.5, numOfClasses + 0.5)
ax[1].set_ylim(0, 1)
ax[1].set_xticklabels(classnames, rotation=0, fontsize=8)



#Plot SNSVTY
bp = ax[2].boxplot(data_SPCFTY, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax[2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax[2].set_axisbelow(True)
ax[2].set_title('Results for Sensitivity')
#ax[2].set_xlabel('Label')
ax[2].set_ylabel('Sensitivity')

# Set the axes ranges and axes labels
ax[2].set_xlim(0.5, numOfClasses + 0.5)
ax[2].set_ylim(0, 1)
ax[2].set_xticklabels(classnames, rotation=0, fontsize=8)



#plt.tight_layout()
plt.show()