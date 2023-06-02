import * as Yup from 'yup'

import {
  AdminDepartmentInterface,
  AdminUserInterface,
  FormAdminSetDoctorProfileValues
} from '../../../interface/AdminInformationInterface'
import { FormRegisterValues } from '../../../interface/ValidateInterface'
import { customFontDoctorProfileTheme } from '../../../themes/authTheme'
import { convertGenderToString } from '../../../utils/convertGenderToString'
import FormikCustomize from '../../../utils/formik/FormikCustomize'
import {
  adminProfileFields,
  adminRegisterProfileFields
} from '../../../utils/formik/formikData'
import {
  NAME_REGEX,
  PASSWORD_REGEX,
  PHONE_REGEX_VN
} from '../../../utils/regex'
import {
  CHECK_ADDRESS_EMPTY,
  CHECK_AGE_EMPTY,
  CHECK_FIRST_NAME_EMPTY,
  CHECK_LAST_NAME_EMPTY,
  CHECK_NAME_EMPTY,
  CHECK_NAME_MATCH_REGEX,
  CHECK_PASSWORD_CONFIRM_EMPTY,
  CHECK_PASSWORD_CONFIRM_MATCH_PASSWORD,
  CHECK_PASSWORD_EMPTY,
  CHECK_PASSWORD_MATCH_REGEX,
  CHECK_PHONE_NUMBER_EMPTY,
  CHECK_PHONE_NUMBER_MATCH_REGEX
} from '../../../utils/validateInform'
import AdminDoctorSettingAvatar from './AdminDoctorSettingAvatar'

type Props = {
  id?: number
  doctorInfor: Partial<AdminUserInterface>
  department: Partial<AdminDepartmentInterface>
  onSubmitDoctorProfile: (values: FormAdminSetDoctorProfileValues) => void
  handleUpdateProfilePictureChange?: (
    selectedFile: string | File
  ) => string | File
}

const AdminDoctorSettings = (props: Props) => {
  const { id, doctorInfor, department, onSubmitDoctorProfile } = props
  const { firstName, lastName, gender, address, phoneNumber, age, email } =
    doctorInfor

  const initialAdminProfileValues = {
    firstName: firstName ? firstName : '',
    lastName: lastName ? lastName : '',
    gender: convertGenderToString(gender),
    age: age ? age.toString() : '',
    address: address ? address : '',
    email: email ? email : '',
    phoneNumber: phoneNumber ? phoneNumber : '',
    departmentId: department.id,
    ...(!id && { password: '' }),
    ...(!id && { confirmPassword: '' })
  }

  const handleProfilesValidationSchema = () => {
    const validateObject = {
      address: Yup.string().trim().required(CHECK_ADDRESS_EMPTY),
      phoneNumber: Yup.string()
        .trim()
        .required(CHECK_PHONE_NUMBER_EMPTY)
        .matches(PHONE_REGEX_VN, CHECK_PHONE_NUMBER_MATCH_REGEX),
      age: Yup.string().trim().required(CHECK_AGE_EMPTY),
      password: Yup.string()
        .trim()
        .required(CHECK_PASSWORD_EMPTY)
        .matches(PASSWORD_REGEX, CHECK_PASSWORD_MATCH_REGEX),
      confirmPassword: Yup.string()
        .trim()
        .required(CHECK_PASSWORD_CONFIRM_EMPTY)
        .oneOf([Yup.ref('password'), ''], CHECK_PASSWORD_CONFIRM_MATCH_PASSWORD)
    }
    if (id) {
      const { password, confirmPassword, ...validateSettingsObject } =
        validateObject
      return Yup.object().shape(validateSettingsObject)
    }
    return Yup.object().shape(validateObject)
  }

  const handleProfilesValidation = (values: FormRegisterValues) => {
    const errors = {
      firstName: '',
      lastName: ''
    }

    errors.firstName = !values.firstName
      ? !values.lastName
        ? CHECK_NAME_EMPTY
        : CHECK_FIRST_NAME_EMPTY
      : !NAME_REGEX.test(values.firstName)
      ? !values.lastName
        ? ''
        : CHECK_NAME_MATCH_REGEX
      : !values.lastName
      ? ''
      : !NAME_REGEX.test(values.lastName)
      ? CHECK_NAME_MATCH_REGEX
      : ''

    errors.lastName = !values.lastName
      ? !values.firstName
        ? ' '
        : CHECK_LAST_NAME_EMPTY
      : !NAME_REGEX.test(values.lastName)
      ? !values.firstName
        ? ''
        : ' '
      : !values.firstName
      ? ''
      : !NAME_REGEX.test(values.lastName)
      ? ' '
      : ''

    const responseErrors = !errors.firstName && !errors.lastName ? {} : errors
    return responseErrors
  }

  return (
    <div className='admin__doctor--setting'>
      <AdminDoctorSettingAvatar
        profilePicture={doctorInfor.profilePicture}
        idDoctor={id}
      />
      <div className='setting__div--formik'>
        <FormikCustomize
          initialFormikValues={initialAdminProfileValues}
          inputFields={id ? adminProfileFields : adminRegisterProfileFields}
          onValidationSchema={handleProfilesValidationSchema}
          onValidate={handleProfilesValidation}
          onSubmitFormik={onSubmitDoctorProfile}
          btnText='Save'
          theme={customFontDoctorProfileTheme}
        />
      </div>
    </div>
  )
}

export default AdminDoctorSettings
