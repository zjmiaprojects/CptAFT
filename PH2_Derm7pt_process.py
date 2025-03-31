import os
import csv
import torch

def Derm7pt(read_file,data_dir, meta_dir, concept):
    n = 0
    with open(read_file, 'a') as file:
        for train_file in os.listdir(data_dir):
            fname = train_file
            # print("fname:",fname)
            if(fname=='0'):
                label=0
                text='Nevus'
            if (fname == '1'):
                label = 1
                text='Melanoma'

            for file_name in os.listdir(data_dir+fname):
            #
                # img_dir=data_dir+fname+'/'+file_name
                # print(img_dir)
                f_name=file_name[0:file_name.find('.')]
                # print('f_name:',f_name)

                with open(meta_dir, 'r') as f:
                    csv_reader = csv.reader(f)
                    for line in csv_reader:
                        concept_lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        concept_text = []
                        # process each line
                        start_index_cli = line[5].find("/")
                        end_index_cli = line[5].find(".")
                        start_index_derm = line[6].find("/")
                        end_index_derm = line[6].find(".")

                        name_scv_derm = line[6][start_index_derm + 1:end_index_derm]
                        name_scv_cli = line[5][start_index_cli + 1:end_index_cli]
                        # print(name_scv)
                        if name_scv_derm==f_name or name_scv_cli==f_name:
                            print(name_scv_derm)
                            pigment_network=line[0]
                            pigment_network_text='pigment_network is '+pigment_network
                            concept_text.append(pigment_network_text)
                            streaks=line[1]
                            streaks_text = 'streaks is ' + streaks
                            concept_text.append(streaks_text)
                            regression_structures = line[2]
                            regression_structures_text = 'regression_structures is ' + regression_structures
                            concept_text.append(regression_structures_text)
                            dots_and_globules = line[3]
                            dots_and_globules_text = 'dots_and_globules is ' + dots_and_globules
                            concept_text.append(dots_and_globules_text)
                            blue_whitish_veil = line[4]
                            blue_whitish_veil_text = 'blue_whitish_veil is ' + blue_whitish_veil
                            concept_text.append(blue_whitish_veil_text)

                            for i in concept_text:
                                # print("i:",i)
                                idx=0
                                for j in concept:
                                    if i==j:
                                        # print("idx:",idx)
                                        concept_lab[idx]=1
                                    idx+=1
                            print("concept_lab:",concept_lab)

                            n+=1
                            img_dir = data_dir + fname + '/' + file_name
                            # img_dir = data_dir + fname + '/' + file_name
                            file.write(f"{img_dir}\t{text}\t{pigment_network}\t{streaks}\t{regression_structures}\t{dots_and_globules}\t{blue_whitish_veil}\t{label}\t{concept_lab}\n")

            print("n:",n)

def PH2D7(read_file, data_dir, meta_dir, concept):
    n = 0
    with open(read_file, 'a') as file:
        for train_file in os.listdir(data_dir):
            fname = train_file

            # print("fname:",fname)
            if(fname=='0'):
                label=0
                text='Nevus'
            if (fname == '1'):
                label = 1
                text='Melanoma'
            #

            for file_name in os.listdir(data_dir+fname):
            #
                # img_dir=data_dir+fname+'/'+file_name
                # print(img_dir)
                f_name=file_name[0:file_name.find('.')]
                # print('f_name:',f_name)

                with open(meta_dir, 'r') as f:
                    csv_reader = csv.reader(f)
                    for line in csv_reader:
                        concept_lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        concept_text = []
                        # process each line
                        start_index_cli = line[5].find("/")
                        end_index_cli = line[5].find(".")
                        start_index_derm = line[6].find("/")
                        end_index_derm = line[6].find(".")

                        name_scv_derm = line[6][start_index_derm + 1:end_index_derm]
                        name_scv_cli = line[5][start_index_cli + 1:end_index_cli]
                        # print(name_scv)
                        if name_scv_derm==f_name or name_scv_cli==f_name:
                            print(name_scv_derm)
                            pigment_network=line[0]
                            pigment_network_text='pigment_network is '+pigment_network
                            concept_text.append(pigment_network_text)
                            streaks=line[1]
                            if streaks=='regular' or streaks=='irregular':
                                streaks = 'present'
                            streaks_text = 'streaks is ' + streaks
                            concept_text.append(streaks_text)
                            regression_structures = line[2]
                            if regression_structures!='absent':
                                regression_structures = 'present'
                            regression_structures_text = 'regression_structures is ' + regression_structures
                            concept_text.append(regression_structures_text)
                            dots_and_globules = line[3]
                            if dots_and_globules=='regular':
                                dots_and_globules = 'typical'
                            if dots_and_globules=='irregular':
                                dots_and_globules = 'atypical'
                            dots_and_globules_text = 'dots_and_globules is ' + dots_and_globules
                            concept_text.append(dots_and_globules_text)
                            blue_whitish_veil = line[4]
                            blue_whitish_veil_text = 'blue_whitish_veil is ' + blue_whitish_veil
                            concept_text.append(blue_whitish_veil_text)

                            for i in concept_text:
                                # print("i:",i)
                                idx=0
                                for j in concept:
                                    if i==j:
                                        # print("idx:",idx)
                                        concept_lab[idx]=1
                                    idx+=1
                            print("concept_lab:",concept_lab)

                            n+=1
                            img_dir = data_dir + fname + '/' + file_name
                            # img_dir = data_dir + fname + '/' + file_name
                            file.write(f"{img_dir}\t{text}\t{pigment_network}\t{streaks}\t{regression_structures}\t{dots_and_globules}\t{blue_whitish_veil}\t{label}\t{concept_lab}\n")

            print("n:",n)


if __name__ == '__main__':
    read_file = "datasets/PH2Derm7pt/test_label_concept.txt"
    data_dir = "datasets/PH2Derm7pt/test/"
    meta_dir = "datasets/Derm7pt/meta/meta.csv"
    # concept=["pigment_network is absent", "pigment_network is typical", "pigment_network is atypical",
    #                     "streaks is absent", "streaks is regular", "streaks is irregular",
    #                     "regression_structures is absent", "regression_structures is combinations",
    #                     "regression_structures is blue areas", "regression_structures is white areas",
    #                     "dots_and_globules is absent", "dots_and_globules is regular","dots_and_globules is irregular",
    #                     "blue_whitish_veil is absent", "blue_whitish_veil is present"]
    concept = ["pigment_network is absent", "pigment_network is typical", "pigment_network is atypical",
               "streaks is absent", "streaks is present",
               "regression_structures is absent", "regression_structures is present",
               "dots_and_globules is absent", "dots_and_globules is typical", "dots_and_globules is atypical",
               "blue_whitish_veil is absent", "blue_whitish_veil is present"]
    PH2D7(read_file, data_dir,meta_dir, concept)