SRC_DIR := src
BUILD_DIR := build
CU_FILES := $(wildcard $(SRC_DIR)/*.cu)
TARGETS := $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%, $(CU_FILES))

NVCC := nvcc
NVCC_FLAGS := -O3 -Xcompiler -fopenmp

all: $(TARGETS)

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)