import AddIcon from '@mui/icons-material/Add'
import {
  Paper,
  Table,
  TableBody,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow
} from '@mui/material'
import { useEffect, useState } from 'react'

import '../../assets/css/pages/adminPage/admin_department_page.css'
import ButtonCustomize from '../../components/ButtonCustomize'
import {
  AdminTableColumn,
  TableDepartment
} from '../../interface/AdminTableInterface'
import { useAppDispatch, useAppSelector } from '../../redux/hooks'
import { getAllDepartmentByAdmin } from '../../redux/thunk/adminThunk/adminDoctorThunk'
import { TableCellProfile } from '../../themes/profileStyle'
import AdminModelCustomize from '../../utils/models/AdminModelCustomize'
import ModalAddDepartmentBody from '../../utils/models/ModalAddDepartmentBody'
import AdminBreadCrumb from './adminBreadcrumb/AdminBreadCrumb'
import AdminDepartmentTableCell from './adminTable/AdminDepartmentTableCell'

const columns: AdminTableColumn[] = [
  { id: 'id', label: '#', minWidth: 70 },
  { id: 'departmentName', label: 'Departments', minWidth: 170 },
  { id: 'actions', label: 'Actions', minWidth: 100, align: 'right' }
]

function createDataTableDepartment(
  id: number,
  departmentName: string,
  departmentPicture: string
): TableDepartment {
  return {
    id,
    departmentName,
    departmentPicture
  }
}

const rowsPerPage = 5

const AdminDepartmentPage = () => {
  const [page, setPage] = useState(0)
  const dispatch = useAppDispatch()
  const { listDepartment } = useAppSelector((state) => state.admin)
  const { isCheckInitialStatus } = useAppSelector((state) => state.auths)
  const [rowsDepartment, setRowsDepartment] = useState<TableDepartment[]>([])

  useEffect(() => {
    if (!isCheckInitialStatus) {
      const fetchData = async () => {
        await dispatch(getAllDepartmentByAdmin())
      }
      fetchData()
    }
  }, [])

  useEffect(() => {
    if (listDepartment) {
      const departments = listDepartment.map((department) => {
        return createDataTableDepartment(
          department.id,
          department.name,
          department.backgroundImage
        )
      })

      setRowsDepartment(departments)
    }
  }, [listDepartment])
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage)
  }
  const [openAdd, setOpenAdd] = useState<boolean>(false)

  const handleOpenAdd = () => setOpenAdd(true)
  const handleCloseAdd = () => setOpenAdd(false)

  return (
    <div className='admin__department'>
      <div className='department__header'>
        <AdminBreadCrumb breadcrumbTitle='Departments' />
        <ButtonCustomize
          text='Add'
          className='btn__department--add'
          onClickBtn={() => handleOpenAdd()}
          icon={<AddIcon />}
        />
      </div>
      <div className='department__body'>
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
              {rowsDepartment
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row, index) => {
                  return (
                    <AdminDepartmentTableCell
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
          count={rowsDepartment.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
        />
      </div>
      <AdminModelCustomize
        classNameHeader='success'
        title='Add Department'
        open={openAdd}
        handleClose={() => handleCloseAdd()}
        bodyModal={
          <ModalAddDepartmentBody
            inputEdit=''
            file=''
            handleClose={() => handleCloseAdd()}
          />
        }
      />
    </div>
  )
}
export default AdminDepartmentPage
