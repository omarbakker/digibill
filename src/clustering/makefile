MAKE = make
MKFLAG = -f


default:
	@echo "List of targets:"
	@echo "ccore_x64        - to build release CCORE library for pyclustering (platform x64)."
	@echo "ccore_x64_clean  - to clean build files of CCORE library (platform x64)."
	@echo "ccore_x86        - to build release CCORE library for pyclustering (platform x86)."
	@echo "ccore_x86_clean  - to clean build files of CCORE library (platform x86)."
	@echo " "
	@echo "clean            - to clean everything."


.PHONY: ccore_x64
ccore_x64:
	$(MAKE) $(MKFLAG) clustering.mk ccore PLATFORM="x64"

.PHONY: ccore_x64_clean
ccore_x64_clean:
	$(MAKE) $(MKFLAG) clustering.mk clean PLATFORM="x64"


.PHONY: ccore_x86
ccore_x86:
	$(MAKE) $(MKFLAG) clustering.mk ccore PLATFORM="x86"


.PHONY: ccore_x86_clean
ccore_x86_clean:
	$(MAKE) $(MKFLAG) clustering.mk clean PLATFORM="x86"

.PHONY: clean
clean:
	$(MAKE) $(MKFLAG) clustering.mk clean PLATFORM="x64"
	$(MAKE) $(MKFLAG) clustering.mk clean PLATFORM="x86"
