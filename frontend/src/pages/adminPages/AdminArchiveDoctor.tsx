import React from 'react'

import AdminDoctorPage from './AdminDoctorPage'

type Props = {}

const AdminArchiveDoctor = (props: Props) => {
  return (
    <>
      <AdminDoctorPage isArchive={1} />
    </>
  )
}

export default AdminArchiveDoctor
