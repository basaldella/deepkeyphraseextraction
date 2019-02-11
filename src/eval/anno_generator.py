import os


def write_anno(output_folder, documents, keyphrases):
    # create output directory if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for doc_id, doc_string in documents.items():

        i = 0
        output_file = open("%s/%s.%s" % (output_folder, doc_id, "ann"), "w",encoding="utf-8")

        for kp in keyphrases[doc_id]:
            kp_string = ' '.join(kp)

            for start_index in list(find_all(doc_string, kp_string)):
                end_index = start_index + len(kp_string)
                output_file.write("T%s\t%s %s %s\t%s\n" %
                                  (i, "NO_TYPE", start_index, end_index, kp_string))

        output_file.close()


def find_all(target_string, substring):
    start = 0
    while True:
        start = target_string.find(substring, start)
        if start == -1: return
        yield start
        start += 1
