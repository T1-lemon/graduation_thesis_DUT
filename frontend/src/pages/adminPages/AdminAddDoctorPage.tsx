import { NavigateBefore, NavigateNext } from '@mui/icons-material'
import { Button, Typography } from '@mui/material'
import { Box, Step, StepLabel, Stepper } from '@mui/material'
import React, { createContext, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import '../../assets/css/pages/adminPage/admin_add_doctor.css'
import ButtonCustomize from '../../components/ButtonCustomize'
import { ProgressListener } from '../../components/Progress'
import { FormAdminSetDoctorProfileValues } from '../../interface/AdminInformationInterface'
import { useAppDispatch } from '../../redux/hooks'
import { getAllDepartmentByAdmin } from '../../redux/thunk/adminThunk/adminDoctorThunk'
import {
  addAdminSetDoctorProfileService,
  checkValidDoctorInforService
} from '../../services/adminServices/adminDoctorService'
import { checkResponseFailed } from '../../utils/checkResponseStatus'
import { converGenderToBoolen } from '../../utils/convertGenderToString'
import {
  QontoConnector,
  QontoStepIcon,
  steps
} from './adminAddDoctor/AdminStep'
import AdminBreadCrumb from './adminBreadcrumb/AdminBreadCrumb'
import AdminDoctorOverviewInput from './adminProfileDoctor/AdminDoctorOverviewInput'
import AdminDoctorOverviewTextArea from './adminProfileDoctor/AdminDoctorOverviewTextArea'
import AdminDoctorSettings from './adminProfileDoctor/AdminDoctorSettings'

const completedSurvey = require('../../assets/img/completed-survey.jpg')

interface DataContextType {
  profilePictureDoctor: File | string
  setProfilePictureDoctor: React.Dispatch<React.SetStateAction<string | File>>
}
export const DataContextAddDoctor = createContext<DataContextType>(
  {} as DataContextType
)

const AdminAddDoctorPage = () => {
  const [activeStep, setActiveStep] = useState<number>(0)
  const [stepFailed, setStepFaild] = useState<number | undefined>()
  const [errorMessage, setErrorMessage] = useState<string>('')
  const [isAllowBack, setIsAllowBack] = useState<boolean>(false)
  const [isAllowNext, setIsAllowNext] = useState<boolean>(false)
  const [profilePictureDoctor, setProfilePictureDoctor] = useState<
    File | string
  >('')
  const [doctorInfor, setDoctorInfor] = useState<
    Partial<FormAdminSetDoctorProfileValues>
  >({})
  const [titleMedical, setTitleMedical] = useState<string>('')
  const [sortDescriptionMedical, setSortDescriptionMedical] =
    useState<string>('')
  const [desscriptionMedical, setDescriptionMedical] = useState<string>('')
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  useEffect(() => {
    const fetchData = async () => {
      await dispatch(getAllDepartmentByAdmin())
    }
    fetchData()
  }, [])

  useEffect(() => {
    if (activeStep === 0) {
      setIsAllowBack(false)
      setIsAllowNext(false)
    }

    if (activeStep === 1) {
      setIsAllowBack(true)
      setIsAllowNext(false)
    }
  }, [activeStep])

  const isStepFailed = (step: number) => {
    return step === stepFailed
  }

  const handleBackStep = () => {
    if (isAllowBack) {
      const backstep = activeStep ? activeStep - 1 : 0
      setActiveStep(backstep)
    }
  }

  const handleNextStep = () => {
    if (isAllowNext) {
      const nextStep = activeStep !== 2 ? activeStep + 1 : 2
      setActiveStep(nextStep)
    }
  }

  const handleChangeSortDescription = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setSortDescriptionMedical(event.target.value)
  }

  const handleChangeDescription = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setDescriptionMedical(event.target.value)
  }

  const handleChangeTitle = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTitleMedical(event.target.value)
  }

  const handleSubmitDoctorProfile = (
    values: FormAdminSetDoctorProfileValues
  ) => {
    const fetchApiCheckValidDoctorInfor = async () => {
      ProgressListener.emit('start')
      const response = await checkValidDoctorInforService(values.email)
      ProgressListener.emit('stop')

      if (checkResponseFailed(response.status)) {
        setErrorMessage('Email already exists')
        setStepFaild(activeStep)
        return
      }
      setDoctorInfor({ ...values, role: 'DOCTOR' })
      const nextStep = activeStep + 1
      setActiveStep(nextStep)
      setStepFaild(undefined)
    }

    if (!profilePictureDoctor) {
      setErrorMessage("You must submit doctor's profile picture")
      setStepFaild(activeStep)
      return
    }
    fetchApiCheckValidDoctorInfor()
  }

  const handleSubmitMedicalExamination = () => {
    const { departmentId, confirmPassword, ...doctor } = doctorInfor
    const dataMedicalExamination = {
      doctor: {
        ...doctor,
        gender: converGenderToBoolen(doctor.gender as string)
      },
      medicalExamination: {
        examinationPrice: 18,
        title: titleMedical,
        shortDescription: sortDescriptionMedical,
        description: desscriptionMedical,
        departmentId: parseInt(doctorInfor.departmentId as string)
      }
    }
    console.log(dataMedicalExamination)

    const formData = new FormData()
    formData.append(
      'data',
      new Blob([JSON.stringify(dataMedicalExamination)], {
        type: 'application/json'
      })
    )
    formData.append('file', profilePictureDoctor)
    const fetchAddAdminSetDoctorProfileApi = async () => {
      ProgressListener.emit('start')
      await addAdminSetDoctorProfileService(formData)
      ProgressListener.emit('stop')
      setActiveStep(activeStep + 1)
    }

    fetchAddAdminSetDoctorProfileApi()
  }

  return (
    <div>
      <div className='admin__doctor--header'>
        <AdminBreadCrumb breadcrumbTitle='Add Doctor' />
      </div>
      <div className='admin__table--body'>
        <div className='admin__doctor--add'>
          <div className='doctor__add--stepper'>
            <Box sx={{ width: '100%' }}>
              <Stepper
                activeStep={activeStep}
                alternativeLabel
                connector={<QontoConnector />}
              >
                {steps.map((label, index) => {
                  const labelProps: {
                    optional?: React.ReactNode
                    error?: boolean
                  } = {}
                  if (isStepFailed(index)) {
                    labelProps.optional = (
                      <Typography variant='caption' color='error'>
                        {errorMessage}
                      </Typography>
                    )
                    labelProps.error = true
                  }

                  return (
                    <Step
                      key={label}
                      className={`${
                        isStepFailed(index) ? 'lable__failed' : 'lable__success'
                      }`}
                    >
                      <StepLabel
                        StepIconComponent={QontoStepIcon}
                        {...labelProps}
                      >
                        {label}
                      </StepLabel>
                    </Step>
                  )
                })}
              </Stepper>
            </Box>
          </div>
          <div className='doctor__add--body'>
            {activeStep === 0 ? (
              <div className='doctor__add--basic'>
                <DataContextAddDoctor.Provider
                  value={{ profilePictureDoctor, setProfilePictureDoctor }}
                >
                  <AdminDoctorSettings
                    doctorInfor={{}}
                    department={{}}
                    onSubmitDoctorProfile={handleSubmitDoctorProfile}
                  />
                </DataContextAddDoctor.Provider>
              </div>
            ) : (
              <></>
            )}
            {activeStep === 1 ? (
              <div className='doctor__add--des'>
                <AdminDoctorOverviewInput onChangeTitle={handleChangeTitle} />
                <AdminDoctorOverviewTextArea
                  onChangeTextArea={handleChangeSortDescription}
                  rows={4}
                  title='Short Description'
                />
                <AdminDoctorOverviewTextArea
                  onChangeTextArea={handleChangeDescription}
                  rows={15}
                  title='Description'
                />
                <div className='doctor__add--btn'>
                  <ButtonCustomize
                    text='Save'
                    className='btn__department--add'
                    onClickBtn={handleSubmitMedicalExamination}
                  />
                </div>
              </div>
            ) : (
              <></>
            )}

            {activeStep === 2 ? (
              <div className='doctor__add--success'>
                <img src={completedSurvey} alt='' className='img--success' />
                <p className='p--success'>Create a successful doctor! 🎉</p>
                <div className='group__div--btn'>
                  <ButtonCustomize
                    text='Go to the list of doctors'
                    className='btn__department--add'
                    onClickBtn={() => {
                      navigate('/admin/doctors/list')
                    }}
                  />
                </div>
              </div>
            ) : (
              <></>
            )}
            <div
              className='div__group--action'
              style={
                activeStep === 2 ? { display: 'none' } : { display: 'flex' }
              }
            >
              <Button
                onClick={() => handleBackStep()}
                variant='contained'
                className='btn__edit btn__group--action'
                startIcon={<NavigateBefore />}
              />
              <Button
                onClick={() => handleNextStep()}
                variant='contained'
                className='btn__edit btn__group--action'
                startIcon={<NavigateNext />}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdminAddDoctorPage
