import {
  Paper,
  Table,
  TableBody,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow
} from '@mui/material'
import React, { useEffect, useState } from 'react'

import { ProgressListener } from '../../components/Progress'
import {
  AdminTableColumn,
  TablePatient
} from '../../interface/AdminTableInterface'
import { useAppDispatch, useAppSelector } from '../../redux/hooks'
import { getAllPatientByAdmin } from '../../redux/thunk/adminThunk/adminPatientThunk'
import { TableCellProfile } from '../../themes/profileStyle'
import AdminBreadCrumb from './adminBreadcrumb/AdminBreadCrumb'
import AdminPatientTableCell from './adminTable/AdminPatientTableCell'

const columns: AdminTableColumn[] = [
  { id: 'patientName', label: 'patients', minWidth: 170 },
  { id: 'gmail', label: 'gmail', minWidth: 170 },
  { id: 'phoneNumber', label: 'phone number', minWidth: 150 },
  { id: 'address', label: 'address', minWidth: 200 },
  { id: 'actions', label: 'Actions', minWidth: 40, align: 'right' }
]

function createDataTablePatient(
  id: number,
  patientName: string,
  profilePicturePatient: string,
  gmail: string,
  phoneNumber: string,
  address: string
): TablePatient {
  return {
    id,
    patientName,
    profilePicturePatient,
    gmail,
    phoneNumber,
    address
  }
}

const rowsPerPage = 6

const AdminPatientPage = () => {
  const [page, setPage] = useState(0)
  const [rowsPatient, setRowsPatient] = useState<TablePatient[]>([])

  const { isCheckInitialStatus } = useAppSelector((state) => state.auths)
  const { listPatient } = useAppSelector((state) => state.admin)
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage)
  }
  const dispatch = useAppDispatch()

  useEffect(() => {
    if (!isCheckInitialStatus) {
      const fetchApiGetAllPatientByAdmin = async () => {
        ProgressListener.emit('start')
        await dispatch(getAllPatientByAdmin())
        ProgressListener.emit('stop')
      }
      fetchApiGetAllPatientByAdmin()
      return
    }
  }, [])

  useEffect(() => {
    if (listPatient !== undefined) {
      const patients = listPatient.map((patient) => {
        return createDataTablePatient(
          patient.id,
          `${patient.firstName} ${patient.lastName}`,
          `${patient.profilePicture}`,
          `${patient.email}`,
          `${patient.phoneNumber}`,
          `${patient.address}`
        )
      })

      setRowsPatient(patients)
    }
  }, [listPatient])

  return (
    <div className='admin__patients--contianer'>
      <div className='admin__table--header'>
        <AdminBreadCrumb breadcrumbTitle='Patients' />
      </div>
      <div className='admin__table--body'>
        <TableContainer component={Paper}>
          <Table sx={{ minWidth: 650 }} aria-label='simple table'>
            <TableHead className='admin__table--head'>
              <TableRow>
                {columns.map((column, index) => (
                  <TableCellProfile
                    key={index}
                    align={column.align}
                    style={{ minWidth: column.minWidth }}
                  >
                    {column.label}
                  </TableCellProfile>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rowsPatient
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row, index) => {
                  return (
                    <AdminPatientTableCell
                      columns={columns}
                      row={row}
                      key={row.id}
                    />
                  )
                })}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[6]}
          component='div'
          count={rowsPatient.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
        />
      </div>
    </div>
  )
}

export default AdminPatientPage
