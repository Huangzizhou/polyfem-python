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
CPMAddPackage("gh:polyfem/polyfem#c18c17c538dd6abc5a7f1b4ed502b7e9bb3a7f51")
