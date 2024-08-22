# Polyfem
# License: MIT

if(TARGET polyfem::polyfem)
    return()
endif()

message(STATUS "Third-party: creating target 'polyfem::polyfem'")

# include(FetchContent)
# FetchContent_Declare(
#     polyfem
#     GIT_REPOSITORY https://github.com/polyfem/polyfem.git
#     GIT_TAG 07ee824f836c445699bbc47ee6f19afbfe39bad4
#     GIT_SHALLOW FALSE
# )
# FetchContent_MakeAvailable(polyfem)

include(CPM)
CPMAddPackage("gh:polyfem/polyfem#81a01a71c1c1a4605812e49da4dad9f0a9404828")
