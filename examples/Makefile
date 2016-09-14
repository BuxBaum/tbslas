ifndef PVFMM_DIR
$(error Cannot find file: PVFMM_DIR)
endif

ifndef TBSLAS_DIR
$(error Cannot find file: TBSLAS_DIR)
endif

-include $(PVFMM_DIR)/MakeVariables

ifndef CXXFLAGS_PVFMM
$(error Cannot find file: MakeVariables)
endif

CUDA_INCLUDES = -I/lrz/sys/parallel/cuda/6.5/sdk/common/inc  #set accordingly (folder with helper_cuda.h and helper_functions.h)

TBSLAS_INC = -I$(TBSLAS_DIR)/src $(CUDA_INCLUDES)
# TBSLAS_LIBS = -ltricubic
# TBSLAS_LDLIBS = -L$(TBSLAS_DIR)/ext/tricubic $(TBSLAS_LIBS)
# CXXFLAGS_TBSLAS = -std=c++0x


RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include
##
CUOBJDIR = ./cudaobj

TARGET_BIN = \
       $(BINDIR)/advection  \
       $(BINDIR)/advection-inverse  \
       $(BINDIR)/advtv  \
       $(BINDIR)/advtvextrap  \
       $(BINDIR)/diffusion \
       $(BINDIR)/advdiff \
       $(BINDIR)/advdiff-ss \
       $(BINDIR)/advdiff-ss-tv \
       $(BINDIR)/advdiff-ss-tv-extrap \
       $(BINDIR)/merge \
       $(BINDIR)/cubic \
       $(BINDIR)/traj \
       $(BINDIR)/field-set \
	$(BINDIR)/example-gaussian \
	$(BINDIR)/chebEval

all : $(TARGET_BIN)

##
HOST_COMPILER = mpic++
NVCC          := $(NVCC_PVFMM)

##
$(BINDIR)/example-gaussian:  $(OBJDIR)/example-gaussian.o  $(CUOBJDIR)/runge-kutta.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS) $(CUDA_LDLIBS_PVFMM) $^ $(LDFLAGS_PVFMM) -o $@
##
$(BINDIR)/chebEval:  $(OBJDIR)/chebEval.o $(CUOBJDIR)/chebEvalCuda.o $(CUOBJDIR)/runge-kutta.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS) $(CUDA_LDLIBS_PVFMM) $^ $(LDFLAGS_PVFMM) -lcusparse -o $@

$(CUOBJDIR)/%.o: $(SRCDIR)/%.cu
	-@$(MKDIRS) $(dir $@)
	$(NVCC) $(NVCCFLAGS_PVFMM) -lineinfo -gencode arch=compute_20,code=sm_20 $(CXXFLAGS_TBSLAS) -I$(INCDIR) $(TBSLAS_INC) -c $^ -o $@   
# add -lineinfo for source file mapping 
# add -maxrregcount for register limiting
# add -Xptxas=-v : show register usage
# set -gencode arch=compute_20,code=sm_20 according to GPU architecture
##


ifeq ($(INTEL_OFFLOAD_OK),yes)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS) -no-offload         $^       $(LDFLAGS_PVFMM) -o $@
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS)                     $^_async $(LDFLAGS_PVFMM) -o $@_async
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS) -D__DEVICE_SYNC__=1 $^_mic   $(LDFLAGS_PVFMM) -o $@_mic

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS) -no-offload         -I$(INCDIR) $(TBSLAS_INC) -c $^ -o $@
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS)                     -I$(INCDIR) $(TBSLAS_INC) -c $^ -o $@_async
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS) -D__DEVICE_SYNC__=1 -I$(INCDIR) $(TBSLAS_INC) -c $^ -o $@_mic

else

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS)                   $^       $(LDFLAGS_PVFMM) $(TBSLAS_LDLIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $(CXXFLAGS_TBSLAS)                 -I$(INCDIR) $(TBSLAS_INC) -c $^ -o $@

endif





clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~
