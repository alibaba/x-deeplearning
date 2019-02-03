/*
 * \file c_api_error.h
 */
#ifndef H_CAPI_ERROR_H_
#define H_CAPI_ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

void Blaze_GetLastErrorString(const char** msg);
void Blaze_SetLastErrorString(const char* format, ...);

#ifdef __cplusplus
}
#endif

#endif  // H_CAPI_ERROR_H_

