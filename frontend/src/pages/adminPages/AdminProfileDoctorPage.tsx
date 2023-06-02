import RemoveIcon from '@mui/icons-material/Remove'
import { Avatar } from '@mui/material'
import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { toast } from 'react-toastify'

import '../../assets/css/pages/adminPage/admin_doctor_profile.css'
import ButtonCustomize from '../../components/ButtonCustomize'
import { ProgressListener } from '../../components/Progress'
import HeaderNav from '../../components/header/HeaderNav'
import {
  AdminDoctorInterface,
  DataAdminSetDoctorProfile,
  FormAdminSetDoctorProfileValues
} from '../../interface/AdminInformationInterface'
import {
  NavListInterface,
  NavListStatus
} from '../../interface/HeaderInterface'
import { useAppDispatch, useAppSelector } from '../../redux/hooks'
import {
  getAllDepartmentByAdmin,
  getDetailDoctorByAdmin,
  updateDoctorProfile
} from '../../redux/thunk/adminThunk/adminDoctorThunk'
import { converGenderToBoolen } from '../../utils/convertGenderToString'
import AdminModelCustomize from '../../utils/models/AdminModelCustomize'
import AdminBreadCrumb from './adminBreadcrumb/AdminBreadCrumb'
import AdminDoctorExperience from './adminProfileDoctor/AdminDoctorExperience'
import AdminDoctorOverview from './adminProfileDoctor/AdminDoctorOverview'
import AdminDoctorReview from './adminProfileDoctor/AdminDoctorReview'
import AdminDoctorSettings from './adminProfileDoctor/AdminDoctorSettings'
import { FooterModalDelete } from './adminTable/AdminDepartmentTableCell'

type Props = {}

const AdminProfileDoctorPage = (props: Props) => {
  const navList: NavListInterface[] = [
    {
      name: 'Overview',
      status: 'overview',
      isActive: true
    },
    {
      name: 'Experience',
      status: 'experience',
      isActive: false
    },
    {
      name: 'Review',
      status: 'review',
      isActive: false
    },
    {
      name: 'Settings',
      status: 'settings',
      isActive: false
    }
  ]

  const [navListActive, setNavListActive] = useState<NavListInterface[]>(
    navList.slice()
  )
  const [doctorState, setDoctorState] = useState<Partial<AdminDoctorInterface>>(
    {}
  )
  const dispatch = useAppDispatch()
  const params = useParams()
  const { doctorDetail } = useAppSelector((state) => state.admin)
  const { doctorInfor, medicalExamination, department } = doctorState
  const id = params.id

  const fetchInformationApi = async () => {
    if (id) {
      ProgressListener.emit('start')
      await dispatch(getDetailDoctorByAdmin(parseInt(id)))
      await dispatch(getAllDepartmentByAdmin())
      ProgressListener.emit('stop')
    }
  }

  useEffect(() => {
    fetchInformationApi()
    setNavListActive(navList)
  }, [])

  useEffect(() => {
    setDoctorState(doctorDetail)
  }, [doctorDetail])

  const handleClickNavItem = (status: NavListStatus) => {
    const navListClone = [...navListActive]
    navListClone.forEach((item) => {
      item.status === status ? (item.isActive = true) : (item.isActive = false)
    })
    setNavListActive(navListClone)
  }

  const [openModel, setOpenModel] = useState<boolean>(false)

  const handleCloseModel = () => {
    setOpenModel(false)
  }

  const hadleOpenModel = () => {
    setOpenModel(true)
  }

  const handleUpdateProfilesSubmit = (
    values: FormAdminSetDoctorProfileValues
  ) => {
    if (id) {
      const fetchApiUpdateProfile = async (
        dataDoctorProfile: DataAdminSetDoctorProfile
      ) => {
        ProgressListener.emit('start')
        await dispatch(updateDoctorProfile(dataDoctorProfile))
        ProgressListener.emit('stop')
        toast.success('Your profile update successfully')
      }

      const dataAdminSetDoctorProfile: DataAdminSetDoctorProfile = {
        doctorId: parseInt(id),
        ...values,
        age: parseInt(values.age),
        gender: converGenderToBoolen(values.gender)
      }

      fetchApiUpdateProfile(dataAdminSetDoctorProfile)
    }
  }

  return (
    <div className='admin__doctor--profile'>
      <div className='admin__table--header'>
        <AdminBreadCrumb breadcrumbTitle='Profile Doctor' />
        <ButtonCustomize
          onClickBtn={hadleOpenModel}
          text='Archive'
          className='btn__department--add btn__archive'
          icon={<RemoveIcon />}
        />
      </div>
      <div className='admin__table--body'>
        <div className='doctor__profile--container'>
          <div className='doctor__profile--header'>
            <div className='profile__box--banner'></div>
            <div className='profile__box--avatar'>
              <Avatar
                src={doctorInfor?.profilePicture}
                className='profile__img'
              />
              <div className='profile__box--des'>
                <div className='profile__p--name'>{`Dr. ${doctorInfor?.firstName} ${doctorInfor?.lastName}`}</div>
                <div className='profile__span--department'>
                  {department?.name}
                </div>
              </div>
            </div>
            <div className='profile__box--nav'>
              <HeaderNav
                navList={navListActive}
                onClickNavItem={handleClickNavItem}
              />
            </div>
          </div>
          <div className='doctor__profile--body'>
            {navListActive.map((item) => {
              if (item.isActive === true && item.status === 'overview') {
                return (
                  <AdminDoctorOverview
                    key={item.name}
                    medicalExamination={medicalExamination}
                  />
                )
              }
              if (item.isActive === true && item.status === 'experience') {
                return (
                  <AdminDoctorExperience
                    key={item.name}
                    medicalExamination={medicalExamination}
                  />
                )
              }
              if (item.isActive === true && item.status === 'review') {
                return <AdminDoctorReview key={item.name} />
              }
              if (item.isActive === true && item.status === 'settings') {
                return (
                  <AdminDoctorSettings
                    key={item.name}
                    id={params.id ? parseInt(params.id) : undefined}
                    doctorInfor={doctorInfor || {}}
                    department={department || {}}
                    onSubmitDoctorProfile={handleUpdateProfilesSubmit}
                  />
                )
              }
              return <div key={item.name}></div>
            })}
          </div>
        </div>
      </div>
      <AdminModelCustomize
        classNameHeader='danger'
        title='Archive Doctor'
        open={openModel}
        handleClose={handleCloseModel}
        titleDelete='You are about to delete this doctor'
        desDelete='This will archive this doctor. Are you sure?'
        footerModal={
          <FooterModalDelete
            doctorId={parseInt(id as string)}
            handleClose={() => handleCloseModel()}
          />
        }
      />
    </div>
  )
}

export default AdminProfileDoctorPage
