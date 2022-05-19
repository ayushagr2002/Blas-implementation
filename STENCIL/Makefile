CC         := icc 
#CFLAGS     := -g -O3 -axCORE-AVX2  -qopenmp
CFLAGS     := -g -O3 -axCORE-AVX2 -qopenmp

OBJECT_FILES := helper.o driver.o stencil.o

stencil: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o stencil

helper.o : helper.c
	$(CC) $(CFLAGS) -c helper.c

driver.o : driver.c 
	$(CC) $(CFLAGS) -c driver.c

stencil.o : stencil.c 
	$(CC) $(CFLAGS) -c stencil.c
