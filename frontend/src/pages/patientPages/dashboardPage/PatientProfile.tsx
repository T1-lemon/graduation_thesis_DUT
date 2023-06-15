import { toast } from 'react-toastify'
import * as Yup from 'yup'

import { ProgressListener } from '../../../components/Progress'
import {
  DataUserProfile,
  FormUserProfileValues
} from '../../../interface/UsersInterface'
import { FormRegisterValues } from '../../../interface/ValidateInterface'
import { useAppDispatch, useAppSelector } from '../../../redux/hooks'
import { updateUserProfile } from '../../../redux/thunk/userThunk'
import { customFontLoginTheme } from '../../../themes/authTheme'
import {
  converGenderToBoolen,
  convertGenderToString
} from '../../../utils/convertGenderToString'
import FormikCustomize from '../../../utils/formik/FormikCustomize'
import { userProfileFields } from '../../../utils/formik/formikData'
import { NAME_REGEX, PHONE_REGEX_VN } from '../../../utils/regex'
import {
  CHECK_ADDRESS_EMPTY,
  CHECK_AGE_EMPTY,
  CHECK_FIRST_NAME_EMPTY,
  CHECK_LAST_NAME_EMPTY,
  CHECK_NAME_EMPTY,
  CHECK_NAME_MATCH_REGEX,
  CHECK_PHONE_NUMBER_EMPTY,
  CHECK_PHONE_NUMBER_MATCH_REGEX
} from '../../../utils/validateInform'

const PatientProfile = () => {
  const dispatch = useAppDispatch()
  const { currentUser } = useAppSelector((state) => state.auths)
  const { id, firstName, lastName, gender, address, phoneNumber, age } =
    currentUser

  const initialUserProfileValues = {
    firstName: firstName ? firstName : '',
    lastName: lastName ? lastName : '',
    gender: convertGenderToString(gender),
    address: address ? address : '',
    phoneNumber: phoneNumber ? phoneNumber : '',
    age: age ? age : null
  }

  const handleUpdateProfilesSubmit = (values: FormUserProfileValues) => {
    const fetchApiUpdateProfile = async (dataUserProfile: DataUserProfile) => {
      ProgressListener.emit('start')
      await dispatch(updateUserProfile(dataUserProfile))
      ProgressListener.emit('stop')
      toast.success('Your profile update successfully')
    }

    const dataUserProfile: DataUserProfile = {
      userId: id,
      ...values,
      gender: converGenderToBoolen(values.gender)
    }

    fetchApiUpdateProfile(dataUserProfile)
  }

  const handleProfilesValidationSchema = () => {
    return Yup.object().shape({
      address: Yup.string().trim().required(CHECK_ADDRESS_EMPTY),
      phoneNumber: Yup.string()
        .trim()
        .required(CHECK_PHONE_NUMBER_EMPTY)
        .matches(PHONE_REGEX_VN, CHECK_PHONE_NUMBER_MATCH_REGEX),
      age: Yup.number().required(CHECK_AGE_EMPTY)
    })
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
    <div className='profile__div--container'>
      <div className='profile__div--header'>
        <p className='profile__p--title'>Profile Settings</p>
        <p className='profile__p--des'>
          Manage your personal information, including phone numbers and email,
          address <span>where you can be contacted</span>
        </p>
        <div className='profile__form'>
          <FormikCustomize
            initialFormikValues={initialUserProfileValues}
            inputFields={userProfileFields}
            onValidationSchema={handleProfilesValidationSchema}
            onValidate={handleProfilesValidation}
            onSubmitFormik={handleUpdateProfilesSubmit}
            btnText='Save'
            theme={customFontLoginTheme}
          />
        </div>
      </div>
    </div>
  )
}

export default PatientProfile
