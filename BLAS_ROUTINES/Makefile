# indicate how the object files are to be created
CC         := icc 
#CFLAGS     := -g -O3 -axCORE-AVX2  -qopenmp
CFLAGS     := -g -O3 -axCORE-AVX2 -qopenmp

OBJECT_FILES := saxpy.o driver.o helper.o sdot.o sscal.o dscal.o ddot.o daxpy.o sgemv.o dgemv.o sgemm.o dgemm.o sger.o dger.o
saxpy: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o saxpy

daxpy: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o daxpy

sgemv: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o sgemv

sger: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o sger

sder: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o sder

dgemv: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o dgemv

sgemm: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o sgemm

dgemm: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o dgemm

ddot: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o ddot

sscal: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o sscal

dscal: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o dscal

sdot: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o sdot

ddot.o: ddot.c
	$(CC) -c $(CFLAGS) ddot.c

sscal.o: sscal.c
	$(CC) -c $(CFLAGS) sscal.c

dscal.o: dscal.c
	$(CC) -c $(CFLAGS) dscal.c

saxpy.o : saxpy.c
	$(CC) $(CFLAGS) -c saxpy.c

daxpy.o : daxpy.c
	$(CC) $(CFLAGS) -c daxpy.c

driver.o : driver.c
	$(CC) $(CFLAGS) -c driver.c

helper.o : helper.c
	$(CC) $(CFLAGS) -c helper.c

sdot.o : sdot.c
	$(CC) $(CFLAGS) -c sdot.c

clean:
	rm *.o 