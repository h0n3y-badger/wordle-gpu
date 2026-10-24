file(READ ${ALLOWED} allowed)
file(READ ${ANSWERS} answers)
file(READ ${NYT} nyt)
file(WRITE ${OUT} "extern const char WORDS_ALLOWED[] = R\"WL(${allowed})WL\";\nextern const char WORDS_ANSWERS[] = R\"WL(${answers})WL\";\nextern const char WORDS_NYT[] = R\"WL(${nyt})WL\";\n")
