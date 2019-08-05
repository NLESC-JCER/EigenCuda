find_package(Boost REQUIRED COMPONENTS unit_test_framework)

list(APPEND test_cases test_dot test_triple_product)

foreach(PROG ${test_cases})
  add_executable(unit_${PROG} ${PROG}.cc)

  target_link_libraries(unit_${PROG} 
    PRIVATE
    eigencuda
    Boost::unit_test_framework)

  target_compile_definitions(unit_${PROG}
  PRIVATE
    BOOST_TEST_DYN_LINK)

  add_test(unit_${PROG} unit_${PROG})
endforeach(PROG)

