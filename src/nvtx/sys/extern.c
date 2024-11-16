#include "wrapper.h"

// Static wrappers

void nvtxInitialize__extern(const void *reserved) { nvtxInitialize(reserved); }
void nvtxDomainMarkEx__extern(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) { nvtxDomainMarkEx(domain, eventAttrib); }
void nvtxMarkEx__extern(const nvtxEventAttributes_t *eventAttrib) { nvtxMarkEx(eventAttrib); }
void nvtxMarkA__extern(const char *message) { nvtxMarkA(message); }
void nvtxMarkW__extern(const wchar_t *message) { nvtxMarkW(message); }
nvtxRangeId_t nvtxDomainRangeStartEx__extern(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) { return nvtxDomainRangeStartEx(domain, eventAttrib); }
nvtxRangeId_t nvtxRangeStartEx__extern(const nvtxEventAttributes_t *eventAttrib) { return nvtxRangeStartEx(eventAttrib); }
nvtxRangeId_t nvtxRangeStartA__extern(const char *message) { return nvtxRangeStartA(message); }
nvtxRangeId_t nvtxRangeStartW__extern(const wchar_t *message) { return nvtxRangeStartW(message); }
void nvtxDomainRangeEnd__extern(nvtxDomainHandle_t domain, nvtxRangeId_t id) { nvtxDomainRangeEnd(domain, id); }
void nvtxRangeEnd__extern(nvtxRangeId_t id) { nvtxRangeEnd(id); }
int nvtxDomainRangePushEx__extern(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) { return nvtxDomainRangePushEx(domain, eventAttrib); }
int nvtxRangePushEx__extern(const nvtxEventAttributes_t *eventAttrib) { return nvtxRangePushEx(eventAttrib); }
int nvtxRangePushA__extern(const char *message) { return nvtxRangePushA(message); }
int nvtxRangePushW__extern(const wchar_t *message) { return nvtxRangePushW(message); }
int nvtxDomainRangePop__extern(nvtxDomainHandle_t domain) { return nvtxDomainRangePop(domain); }
int nvtxRangePop__extern(void) { return nvtxRangePop(); }
nvtxResourceHandle_t nvtxDomainResourceCreate__extern(nvtxDomainHandle_t domain, nvtxResourceAttributes_t *attribs) { return nvtxDomainResourceCreate(domain, attribs); }
void nvtxDomainResourceDestroy__extern(nvtxResourceHandle_t resource) { nvtxDomainResourceDestroy(resource); }
void nvtxDomainNameCategoryA__extern(nvtxDomainHandle_t domain, uint32_t category, const char *name) { nvtxDomainNameCategoryA(domain, category, name); }
void nvtxDomainNameCategoryW__extern(nvtxDomainHandle_t domain, uint32_t category, const wchar_t *name) { nvtxDomainNameCategoryW(domain, category, name); }
void nvtxNameCategoryA__extern(uint32_t category, const char *name) { nvtxNameCategoryA(category, name); }
void nvtxNameCategoryW__extern(uint32_t category, const wchar_t *name) { nvtxNameCategoryW(category, name); }
void nvtxNameOsThreadA__extern(uint32_t threadId, const char *name) { nvtxNameOsThreadA(threadId, name); }
void nvtxNameOsThreadW__extern(uint32_t threadId, const wchar_t *name) { nvtxNameOsThreadW(threadId, name); }
nvtxStringHandle_t nvtxDomainRegisterStringA__extern(nvtxDomainHandle_t domain, const char *string) { return nvtxDomainRegisterStringA(domain, string); }
nvtxStringHandle_t nvtxDomainRegisterStringW__extern(nvtxDomainHandle_t domain, const wchar_t *string) { return nvtxDomainRegisterStringW(domain, string); }
nvtxDomainHandle_t nvtxDomainCreateA__extern(const char *name) { return nvtxDomainCreateA(name); }
nvtxDomainHandle_t nvtxDomainCreateW__extern(const wchar_t *name) { return nvtxDomainCreateW(name); }
void nvtxDomainDestroy__extern(nvtxDomainHandle_t domain) { nvtxDomainDestroy(domain); }
void nvtxNameCuDeviceA__extern(CUdevice device, const char *name) { nvtxNameCuDeviceA(device, name); }
void nvtxNameCuDeviceW__extern(CUdevice device, const wchar_t *name) { nvtxNameCuDeviceW(device, name); }
void nvtxNameCuContextA__extern(CUcontext context, const char *name) { nvtxNameCuContextA(context, name); }
void nvtxNameCuContextW__extern(CUcontext context, const wchar_t *name) { nvtxNameCuContextW(context, name); }
void nvtxNameCuStreamA__extern(CUstream stream, const char *name) { nvtxNameCuStreamA(stream, name); }
void nvtxNameCuStreamW__extern(CUstream stream, const wchar_t *name) { nvtxNameCuStreamW(stream, name); }
void nvtxNameCuEventA__extern(CUevent event, const char *name) { nvtxNameCuEventA(event, name); }
void nvtxNameCuEventW__extern(CUevent event, const wchar_t *name) { nvtxNameCuEventW(event, name); }
