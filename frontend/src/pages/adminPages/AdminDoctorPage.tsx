import { Add, Person, SearchOff } from '@mui/icons-material'
import { TablePagination } from '@mui/material'
import Grid from '@mui/material/Grid'
import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import '../../assets/css/pages/adminPage/admin_doctor_page.css'
import ButtonCustomize from '../../components/ButtonCustomize'
import { ProgressListener } from '../../components/Progress'
import { AdminDoctorInterface } from '../../interface/AdminInformationInterface'
import { TableDoctor } from '../../interface/AdminTableInterface'
import { useAppSelector } from '../../redux/hooks'
import { getAllDoctorByAdminService } from '../../services/adminServices/adminDoctorService'
import AdminBreadCrumb from './adminBreadcrumb/AdminBreadCrumb'

function createDataTableDoctor(
  id: number,
  doctorname: string,
  departmentName: string,
  profilePictureDoctor?: string
): TableDoctor {
  return {
    id,
    doctorname,
    departmentName,
    profilePictureDoctor
  }
}

const rowsPerPage = 10
type Props = {
  isArchive: number
}

function AdminDoctorPage(props: Props) {
  const { isArchive } = props
  const navigate = useNavigate()

  const [page, setPage] = useState(0)
  const { isCheckInitialStatus } = useAppSelector((state) => state.auths)
  const [listDoctor, setListDoctor] = useState<AdminDoctorInterface[]>([])
  const [rowsDoctor, setRowsDoctor] = useState<TableDoctor[]>([])

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage)
  }

  useEffect(() => {
    if (!isCheckInitialStatus) {
      const fetchApiGetAllDoctorByAdmin = async () => {
        ProgressListener.emit('start')
        const responses = await getAllDoctorByAdminService(isArchive)
        const listDoctor: AdminDoctorInterface[] = responses.map(
          (response: any) => {
            return {
              doctorInfor: response.doctor,
              medicalExamination: response.medicalExamination,
              department: response.medicalExamination.department
            }
          }
        )
        setListDoctor(listDoctor)
        ProgressListener.emit('stop')
      }
      fetchApiGetAllDoctorByAdmin()
      return
    }
  }, [])

  useEffect(() => {
    console.log(listDoctor.length)
    if (listDoctor.length !== 0) {
      const doctors = listDoctor.map((doctor) => {
        const { doctorInfor, department } = doctor
        return createDataTableDoctor(
          doctorInfor.id,
          `Dr. ${doctorInfor.lastName}`,
          `${department.name}`,
          `${doctorInfor.profilePicture}`
        )
      })
      setRowsDoctor(doctors)
    }
  }, [listDoctor])

  return (
    <div className='admin__doctors--container'>
      <div className='admin__table--header'>
        <AdminBreadCrumb breadcrumbTitle='Doctors' />
        {isArchive ? (
          <></>
        ) : (
          <ButtonCustomize
            text='Add'
            className='btn__department--add'
            onClickBtn={() => navigate('/admin/doctors/add')}
            icon={<Add />}
          />
        )}
      </div>
      {rowsDoctor.length === 0 ? (
        <div className='admin__table--body'>
          <div className='admin__doctor--empty'>
            <SearchOff className='empty__img' />
            <p className='empty__p'>There are no doctors present in here.</p>
          </div>
        </div>
      ) : (
        <div className='admin__table--body'>
          <Grid
            container
            spacing={2}
            direction='row'
            justifyContent='space-around'
            className='admin__grid--doctor'
          >
            {rowsDoctor
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row, index) => {
                return (
                  <Grid
                    item
                    xs={2.3}
                    key={row.id}
                    className={`doctor__div--box doctor__row--${index}`}
                  >
                    <div
                      className='doctor__div--container'
                      onClick={
                        isArchive
                          ? () => {}
                          : () => navigate(`/admin/doctors/profile/${row.id}`)
                      }
                    >
                      <div className='doctor__div--header'>
                        {row.profilePictureDoctor !== 'null' ? (
                          <img
                            src={row.profilePictureDoctor}
                            alt='#'
                            className='doctor__img'
                          />
                        ) : (
                          <div className='doctor__placeholder'>
                            <Person className='doctor__placeholder--icon' />
                          </div>
                        )}
                      </div>
                      <div className='doctor__div--body'>
                        <p className='doctor__p--name'>{row.doctorname}</p>
                        <span className='doctor__span--department'>
                          {row.departmentName}
                        </span>
                      </div>
                    </div>
                  </Grid>
                )
              })}
            {rowsPerPage * (page + 1) > rowsDoctor.length ? (
              <>
                {[...Array(rowsPerPage * (page + 1) - rowsDoctor.length)].map(
                  (_, index) => (
                    <Grid key={index} item xs={2.3}></Grid>
                  )
                )}
              </>
            ) : (
              <></>
            )}
          </Grid>
          <TablePagination
            rowsPerPageOptions={[6]}
            component='div'
            count={rowsDoctor.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
          />
        </div>
      )}
    </div>
  )
}

export default AdminDoctorPage
