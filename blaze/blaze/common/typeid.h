/*
 * \file typeid.h
 * \desc The typeid utility.
 */
#pragma once

#include <string>
#include <type_traits>
#include <typeinfo>

namespace blaze {

// A utility function to demangle a function name
std::string Demangle(const char* name);

template <typename T>
static const char* DemangleType() {
#ifdef __GXX_RTTI
  static const std::string name = Demangle(typeid(T).name());
  return name.c_str();
#else
  return "(RTTI disabled, cannot show name)";
#endif
}

/*!
 * \brief whether a type is pod type
 * \tparam T the type to query
 */
template<typename T>
struct is_pod {
  /*! \brief the value of the traits */
  static const bool value = std::is_pod<T>::value;
};

/*!
 * \brief whether a type is integer type
 * \tparam T the type to query
 */
template<typename T>
struct is_integral {
  /*! \brief the value of the traits */
  static const bool value = std::is_integral<T>::value;
};

/*!
 * \brief whether a type is floating point type
 * \tparam T the type to query
 */
template<typename T>
struct is_floating_point {
  /*! \brief the value of the traits */
  static const bool value = std::is_floating_point<T>::value;
};

/*!
 * \brief whether a type is arithemetic type
 * \tparam T the type to query
 */
template<typename T>
struct is_arithmetic {
  /*! \brief the value of the traits */
  static const bool value = std::is_arithmetic<T>::value;
};

/*!
 * \brief helper class to construct a string that represents type name
 *
 * Specialized this class to defined type name of custom types
 *
 * \tparam T the type to query
 */
template<typename T>
struct type_name_helper {
  /*!
   * \return a string of typename.
   */
  static inline std::string value() {
    return "";
  }
};

/*!
 * \brief the string representation of type name
 * \tparam T the type to query
 * \return a const string of typename.
 */
template<typename T>
inline std::string type_name() {
  return type_name_helper<T>::value();
}

/*!
 * \brief whether a type have save/load function
 * \tparam T the type to query
 */
template<typename T>
struct has_saveload {
  /*! \brief the value of the traits */
  static const bool value = false;
};

/*!
 * \brief template to select type based on condition
 * For example, IfThenElseType<true, int, float>::Type will give int
 * \tparam cond the condition
 * \tparam Then the typename to be returned if cond is true
 * \tparam Else typename to be returned if cond is false
*/
template<bool cond, typename Then, typename Else>
struct IfThenElseType;

/*! \brief macro to quickly declare traits information */
#define DLAZE_DECLARE_TRAITS(Trait, Type, Value)      \
  template<>                                          \
  struct Trait<Type> {                                \
    static const bool value = Value;                  \
  }

/*! \brief macro to quickly declare traits information */
#define DLAZE_DECLARE_TYPE_NAME(Type, Name)           \
  template<>                                          \
  struct type_name_helper<Type> {                     \
    static inline std::string value() {               \
      return Name;                                    \
    }                                                 \
  }

DLAZE_DECLARE_TYPE_NAME(float, "float");
DLAZE_DECLARE_TYPE_NAME(double, "double");
DLAZE_DECLARE_TYPE_NAME(int, "int");
DLAZE_DECLARE_TYPE_NAME(uint32_t, "int (non-negative)");
DLAZE_DECLARE_TYPE_NAME(uint64_t, "long (non-negative)");
DLAZE_DECLARE_TYPE_NAME(std::string, "string");
DLAZE_DECLARE_TYPE_NAME(bool, "boolean");
DLAZE_DECLARE_TYPE_NAME(void*, "ptr");

template<typename Then, typename Else>
struct IfThenElseType<true, Then, Else> {
  typedef Then Type;
};

template<typename Then, typename Else>
struct IfThenElseType<false, Then, Else> {
  typedef Else Type;
};

}  // namespace blaze
