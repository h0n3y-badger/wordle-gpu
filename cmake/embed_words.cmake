file(READ ${ALLOWED} allowed)
file(READ ${ANSWERS} answers)
file(WRITE ${OUT} "extern const char WORDS_ALLOWED[] = R\"WL(${allowed})WL\";\nextern const char WORDS_ANSWERS[] = R\"WL(${answers})WL\";\n")
