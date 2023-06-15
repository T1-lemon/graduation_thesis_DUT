import PersonIcon from '@mui/icons-material/Person'
import PhoneIcon from '@mui/icons-material/Phone'
import {
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup
} from '@mui/material'
import InputAdornment from '@mui/material/InputAdornment'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import TextField from '@mui/material/TextField'
import { error } from 'console'
import { useFormik } from 'formik'
import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { toast } from 'react-toastify'
import { Client } from 'stompjs'
import * as Yup from 'yup'

import ButtonCustomize from '../../../components/ButtonCustomize'
import ModalConfirm from '../../../components/ModalConfirm'
import { ITimeSlotResponse } from '../../../interface/TimeSlotInterfaces'
import {
  DataUserProfile,
  UserInformation
} from '../../../interface/UsersInterface'
import { useAppDispatch, useAppSelector } from '../../../redux/hooks'
import { getAllMedicalExaminationTimeThunk } from '../../../redux/slices/medicalExaminationSlice'
import { getListAppointment } from '../../../redux/thunk/appointmentThunk'
import { updateUserProfile } from '../../../redux/thunk/userThunk'
import { bookAppointmentService } from '../../../services/appointmentService'
import { MODAL_ACTION_CONFIRM } from '../../../utils/contant'
import { PHONE_REGEX_VN } from '../../../utils/regex'
import {
  CHECK_ADDRESS_EMPTY,
  CHECK_FIRST_NAME_EMPTY,
  CHECK_LAST_NAME_EMPTY,
  CHECK_PHONE_NUMBER_EMPTY,
  CHECK_PHONE_NUMBER_MATCH_REGEX
} from '../../../utils/validateInform'
import CustomizedSwitch from './CustomizedSwitch'

const flagVN = require('../../../assets/img/vietnam_flag.png')

type IPatientInformation = {
  firstName?: string
  lastName?: string
  phoneNumber?: string
  address?: string
  gender: number
  [key: string]: any
}
type Props = {
  timeSlotResponse?: ITimeSlotResponse
  stompClient?: Client
}

export default function FormAppointment(props: Props) {
  const { timeSlotResponse, stompClient } = props
  const { currentUser } = useAppSelector((state) => state.auths)
  const [checkSwitch, setCheckSwitch] = useState(false)
  const navigate = useNavigate()
  const dispatch = useAppDispatch()

  const formik = useFormik<IPatientInformation>({
    initialValues: {
      firstName: currentUser.firstName,
      lastName: currentUser.lastName,
      phoneNumber: currentUser.phoneNumber,
      address: currentUser.address,
      gender: currentUser.gender ? 1 : 0
    },

    validationSchema: Yup.object().shape({
      firstName: Yup.string().required(CHECK_FIRST_NAME_EMPTY),
      lastName: Yup.string().required(CHECK_LAST_NAME_EMPTY),
      phoneNumber: Yup.string()
        .required(CHECK_PHONE_NUMBER_EMPTY)
        .matches(PHONE_REGEX_VN, CHECK_PHONE_NUMBER_MATCH_REGEX),
      address: Yup.string().required(CHECK_ADDRESS_EMPTY)
    }),

    onSubmit: (values) => {
      const inputData: DataUserProfile = {
        ...values,
        gender: +values.gender === 1,
        userId: currentUser.id
      }

      if (checkSwitch) {
        updateProfile(inputData)
      }

      bookAppointmentSocket()
      bookAppointmentApi()
      navigate('/user/appointments')
    }
  })

  const { values, errors, handleChange, handleSubmit, handleBlur } = formik

  const bookAppointmentApi = async () => {
    await bookAppointmentService(
      timeSlotResponse?.doctorId as number,
      timeSlotResponse?.timeSlotDTO.id as number
    )
    await dispatch(getAllMedicalExaminationTimeThunk())
    await dispatch(getListAppointment())
    toast.success('Your appointment has been booked successfully.')
  }

  const bookAppointmentSocket = () => {
    if (stompClient) {
      let appointment = {
        patientId: currentUser.id,
        doctorId: timeSlotResponse?.doctorId,
        avatarPatient: currentUser.profilePicture,
        startTime: timeSlotResponse?.timeSlotDTO.startTime,
        duration: timeSlotResponse?.timeSlotDTO.duration,
        patientName: currentUser.firstName + ' ' + currentUser.lastName,
        isRead: false,
        createdDate: new Date()
      }
      stompClient.send(
        '/app/private-appointment',
        {},
        JSON.stringify(appointment)
      )
    }
  }

  const updateProfile = async (dataUserProfile: DataUserProfile) => {
    await dispatch(updateUserProfile(dataUserProfile))
  }

  const handleChangeSwitch = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.checked) {
      values.firstName = currentUser.firstName
      values.lastName = currentUser.lastName
      values.gender = currentUser.gender ? 1 : 0
      values.phoneNumber = currentUser.phoneNumber
      values.address = currentUser.address

      Object.keys(errors).forEach((key: any) => {
        errors[key] = ''
      })
    }
    setCheckSwitch(event.target.checked)
  }

  const [openModalConfirm, setOpenModalConfirm] = useState(false)
  const toggleModalConfirm = () => {
    setOpenModalConfirm(!openModalConfirm)
  }

  const handleBookAppointment = async (type: string) => {
    if (type === MODAL_ACTION_CONFIRM) {
      handleSubmit()
    }
    toggleModalConfirm()
  }

  return (
    <>
      <div className='container__form__book box__personal__infor'>
        <div className='title__box'>
          <h3 className='form__title'>Personal Information</h3>
        </div>

        <div className='inner__box'>
          <div className='container__switch'>
            <CustomizedSwitch
              handleChange={handleChangeSwitch}
              checkSwitch={checkSwitch}
            />
            {checkSwitch ? (
              <p className='text__notification'>
                If you change your information, your personal information will
                be updated
              </p>
            ) : (
              ''
            )}
          </div>
          <form action='' onSubmit={handleSubmit}>
            <div className='container__input'>
              <label className='label__input'>
                First Name <span className='sign__required'>*</span>
              </label>
              <TextField
                disabled={!checkSwitch}
                fullWidth
                className='text__field'
                onBlur={handleBlur}
                onChange={handleChange}
                placeholder='e.g. Do Van'
                error={errors.firstName ? true : false}
                helperText={errors.firstName}
                name='firstName'
                value={values.firstName}
              />
            </div>

            <div className='container__input'>
              <label className='label__input'>
                Last Name <span className='sign__required'>*</span>
              </label>
              <TextField
                disabled={!checkSwitch}
                fullWidth
                className='text__field'
                onBlur={handleBlur}
                onChange={handleChange}
                placeholder='e.g. Duc Thanh'
                error={errors.lastName ? true : false}
                helperText={errors.lastName}
                name='lastName'
                value={values.lastName}
              />
            </div>

            <div className='container__input'>
              <label className='label__input'>
                Gender <span className='sign__required'>*</span>
              </label>
              <RadioGroup
                aria-labelledby='demo-radio-buttons-group-label'
                // defaultValue={0}
                name='gender'
                onChange={handleChange}
                value={values.gender}
              >
                <FormControlLabel
                  value={0}
                  control={<Radio size='small' />}
                  label='Female'
                  className='form__control__label'
                  disabled={!checkSwitch}
                />
                <FormControlLabel
                  value={1}
                  control={<Radio size='small' />}
                  label='Male'
                  className='form__control__label'
                  disabled={!checkSwitch}
                />
              </RadioGroup>
            </div>

            <div className='container__input'>
              <label className='label__input'>
                Phone Number <span className='sign__required'>*</span>
              </label>
              <TextField
                disabled={!checkSwitch}
                fullWidth
                className='text__field phone__input'
                onBlur={handleBlur}
                onChange={handleChange}
                placeholder='e.g. 0968212841'
                error={errors.phoneNumber ? true : false}
                helperText={errors.phoneNumber}
                name='phoneNumber'
                value={values.phoneNumber}
                type='text'
                InputProps={{
                  startAdornment: (
                    <InputAdornment position='start'>
                      <img className='img__flag' src={flagVN} alt='' />
                    </InputAdornment>
                  )
                }}
              />
            </div>

            <div className='container__input'>
              <label className='label__input'>
                Address <span className='sign__required'>*</span>
              </label>
              <TextField
                disabled={!checkSwitch}
                fullWidth
                className='text__field'
                onBlur={handleBlur}
                onChange={handleChange}
                placeholder='e.g. 48 Doan Van Cu, Lien Chieu District, Da Nang City'
                error={errors.address ? true : false}
                helperText={errors.address}
                name='address'
                value={values.address}
              />
            </div>
          </form>

          <ButtonCustomize
            text='Book an appointment'
            onClickBtn={() => {
              if (Object.keys(errors).length === 0 || !checkSwitch)
                toggleModalConfirm()
            }}
          />
        </div>
      </div>

      <ModalConfirm
        openModalConfirm={openModalConfirm}
        onAction={handleBookAppointment}
        title='Book Appointment'
        textBtn='Create'
        backgroundColorBtnConfirm='var(--primary-color)'
        contentBody={
          <>
            <h2 style={{ padding: '20px 0px' }}>Are you sure ?</h2>
            <p>Do you really want to book appointment?</p>
            {checkSwitch ? (
              <p style={{ color: '#da4040' }}>
                Your personal information will be updated.
              </p>
            ) : (
              ''
            )}
          </>
        }
      />
    </>
  )
}
